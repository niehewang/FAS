#!/usr/bin/env python3
"""Train one controlled sibling expert with PEFT LoRA/QLoRA.

The script intentionally keeps the training recipe in YAML so every expert can
share exactly the same optimization hyperparameters while changing only the
training corpus/domain. It supports:
  * Hugging Face dataset IDs or local json/jsonl files;
  * a single `text_field`, or prompt/response fields with a shared template;
  * full precision / bf16 LoRA or 4-bit QLoRA;
  * deterministic metadata export for lineage manifests.

Example:
  python scripts/train_lora_expert.py --config configs/experts/math.yaml

No dataset/model is downloaded until this script is explicitly executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_cfg(path: str) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    return cfg


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_text(example: dict, data_cfg: dict) -> str:
    """Build a plain supervised training string."""
    text_field = data_cfg.get("text_field")
    if text_field:
        return str(example[text_field])
    pfield = data_cfg.get("prompt_field", "prompt")
    rfield = data_cfg.get("response_field", "response")
    template = data_cfg.get("template", "{prompt}\n{response}")
    return template.format(prompt=str(example[pfield]), response=str(example[rfield]))


def build_prompt_and_full(example: dict, data_cfg: dict, tok) -> tuple[str, str]:
    """Return (prompt_prefix, full_text) for response-only loss masking.

    If `use_chat_template: true`, tokenizer.apply_chat_template is used with a
    user message and assistant answer; otherwise `prompt_template` and
    `template` are used. `text_field` examples cannot be prompt-masked because
    their prompt/response boundary is unknown.
    """
    if data_cfg.get("text_field"):
        full = str(example[data_cfg["text_field"]])
        return "", full
    pfield = data_cfg.get("prompt_field", "prompt")
    rfield = data_cfg.get("response_field", "response")
    prompt = str(example[pfield]); response = str(example[rfield])
    if data_cfg.get("use_chat_template", False):
        system = data_cfg.get("system_prompt")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": str(system)})
        msgs.append({"role": "user", "content": prompt})
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_msgs = msgs + [{"role": "assistant", "content": response}]
        full = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        return prefix, full
    prompt_template = data_cfg.get("prompt_template", "{prompt}\n")
    full_template = data_cfg.get("template", "{prompt}\n{response}")
    return prompt_template.format(prompt=prompt), full_template.format(prompt=prompt, response=response)


class SupervisedPaddingCollator:
    """Pad input_ids/attention_mask while preserving precomputed -100 labels."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded = []
        pad_left = self.tokenizer.padding_side == "left"
        for lab in labels:
            n = max_len - len(lab)
            padded.append(([-100] * n + lab) if pad_left else (lab + [-100] * n))
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # Imports are delayed so `--help` and repository tests do not require GPU deps.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    model_name = cfg["model_name"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    if data_cfg.get("path"):
        ds = load_dataset("json", data_files=data_cfg["path"], split=data_cfg.get("split", "train"))
    else:
        ds = load_dataset(data_cfg["name"], data_cfg.get("config_name"), split=data_cfg.get("split", "train"))
    if data_cfg.get("max_examples"):
        ds = ds.select(range(min(len(ds), int(data_cfg["max_examples"]))))

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(cfg.get("trust_remote_code", True)))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_len = int(cfg.get("max_seq_length", 2048))

    mask_prompt_loss = bool(data_cfg.get("mask_prompt_loss", True)) and not bool(data_cfg.get("text_field"))

    def tokenize(ex):
        prefix, text = build_prompt_and_full(ex, data_cfg, tok)
        z = tok(text, truncation=True, max_length=max_len, padding=False, add_special_tokens=True)
        labels = list(z["input_ids"])
        if mask_prompt_loss and prefix:
            pz = tok(prefix, truncation=True, max_length=max_len, padding=False, add_special_tokens=True)
            prompt_len = min(len(pz["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            # If truncation removed the full answer, this example has no supervised token.
            if all(x == -100 for x in labels):
                labels[-1] = z["input_ids"][-1]
        z["labels"] = labels
        return z

    remove_cols = ds.column_names
    ds_tok = ds.map(tokenize, remove_columns=remove_cols, desc="tokenize")

    qlora = bool(cfg.get("qlora_4bit", False))
    quant_cfg = None
    if qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_double_quant", True)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        quantization_config=quant_cfg,
        device_map=cfg.get("device_map", "auto"),
    )
    if qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules"),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    if bool(cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_cfg = cfg["training"]
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=float(train_cfg.get("epochs", 1.0)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=int(train_cfg.get("save_steps", 250)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        optim=train_cfg.get("optim", "paged_adamw_8bit" if qlora else "adamw_torch"),
        report_to=train_cfg.get("report_to", "none"),
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
    )
    collator = SupervisedPaddingCollator(tok)
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tok, data_collator=collator)
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))

    meta = {
        "model_name": model_name,
        "domain": cfg.get("domain"),
        "seed": seed,
        "config_sha256": stable_hash(cfg),
        "num_train_examples": len(ds_tok),
        "output_dir": str(out),
        "adapter_dir": str(out / "adapter"),
        "qlora_4bit": qlora,
        "mask_prompt_loss": mask_prompt_loss,
        "use_chat_template": bool(data_cfg.get("use_chat_template", False)),
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch": torch.__version__,
        },
    }
    (out / "training_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
