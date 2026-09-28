#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def fmt_gb(x: float) -> str:
    return f"{x:.1f} GB"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/resource_budget.yaml'))
    ap.add_argument('--output', default=str(ROOT / 'RESOURCE_BUDGET.md'))
    args = ap.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    m, t, a, s = c['models'], c['training'], c['audit'], c['storage']

    base_cache = sum(v['hf_bf16_gb'] for v in m.values())
    adapter_disk = t['formal_expert_count'] * 0.12 * s['adapter_checkpoint_multiplier']
    student_disk = (t['core_l5_student_count'] + t['core_l7_student_count']) * 0.08 * s['adapter_checkpoint_multiplier']
    materialized = s['keep_materialized_4b_merges_simultaneously'] * m['qwen3_4b_base']['hf_bf16_gb']
    misc = 25.0
    working = base_cache + adapter_disk + student_disk + materialized + misc
    safe = working * 1.5

    expert_tokens = t['formal_expert_count'] * t['expert_examples_per_domain'] * t['avg_train_tokens_per_example']
    kd_tokens = (t['core_l5_student_count'] + t['core_l7_student_count']) * t['kd_examples_per_student'] * t['avg_train_tokens_per_example']
    sft_tokens = 3 * t['deep_sft_examples_per_seed'] * t['avg_train_tokens_per_example']
    total_tokens = expert_tokens + kd_tokens + sft_tokens
    B, samples = a['default_selected_B'], a['stochastic_samples_m']
    online_calls = 2 * B * samples

    lines = [
        '# GPU / Storage Planning Budget (v0.9)', '',
        'This is a **planning document, not a measured benchmark**. Counts in `configs/resource_budget.yaml` are placeholders until the final training corpora are audited.', '',
        '## Core execution set', '',
        '- 12 independent Qwen3-4B sibling QLoRA experts = 3 genealogy seeds × 4 domains.',
        '- 3 core 1.7B Mixture-KD students.',
        '- 3 core 1.7B Deep-Ancestry students.',
        '- G0 calibration descendants are weighted-adapter compositions and require no new 4B training.',
        '- 0.6B, Router-KD, second model family and diffusion remain conditional/optional.', '',
        '## Storage', '',
        '| Component | Planning allowance |', '|---|---:|',
        f"| Shared base-model cache (4B + 1.7B + 0.6B) | {fmt_gb(base_cache)} |",
        f"| 12 formal 4B LoRA training outputs/checkpoints | {fmt_gb(adapter_disk)} |",
        f"| 6 core 1.7B student LoRA outputs/checkpoints | {fmt_gb(student_disk)} |",
        f"| Up to {s['keep_materialized_4b_merges_simultaneously']} materialized 4B merges at once | {fmt_gb(materialized)} |",
        f"| Response banks / teacher data / embeddings / logs / scratch | {fmt_gb(misc)} |",
        f"| **Estimated active working set** | **{fmt_gb(working)}** |",
        f"| **Recommended free disk (×1.5 safety margin)** | **{fmt_gb(safe)}** |", '',
        'Do not materialize every merge permanently. Keep LoRA adapters + genealogy manifests as canonical artifacts; materialize full 4B checkpoints only when required, record hashes, then delete/rebuild disposable intermediates.', '',
        '## Training-token planning', '',
        f"- Expert fine-tuning: ~{expert_tokens/1e6:.1f}M sequence tokens.",
        f"- Core student distillation: ~{kd_tokens/1e6:.1f}M sequence tokens.",
        f"- Deep-chain auxiliary SFT: ~{sft_tokens/1e6:.1f}M sequence tokens.",
        f"- **Total planned train-token exposure:** ~{total_tokens/1e6:.1f}M tokens.", '',
        'These values are scheduling placeholders; the paper must report audited realized counts.', '',
        '## GPU memory policy', '',
        'Formal training uses 4-bit QLoRA, gradient checkpointing, micro-batch 1, grad accumulation 16, max sequence length 2048. A 24 GB GPU is a practical *planning floor* for the 4B QLoRA jobs, not a guarantee. Prefer 40–80 GB GPUs when available. Multi-teacher response generation is sequential-by-teacher, so four 4B teachers never need to coexist in VRAM.', '',
        'Before launching the 12 formal experts, run one Math expert for 20–50 optimizer steps and record peak allocated VRAM, tokens/s and wall time. Repeat once for a 1.7B student. Use those measured throughputs for cluster-time estimates rather than parameter-count extrapolation.', '',
        '## Online audit budget', '',
        f"For default B={B}, m={samples}: `2 × B × m = {online_calls}` generations per audited model.",
        'Report this separately from offline ancestor-bank construction. LLM-DNA and modelDNA retain their own access accounting (`prompts` and `weights`).', '',
        '## Stop/go rule', '',
        '1. Synthetic + pilot geometry.',
        '2. Train the 12 formal experts and run L1/L2; stop if active IFRF does not improve conditioning/recovery.',
        '3. Then run the 3 × 1.7B Mixture-KD students.',
        '4. Only if KD ancestry is recoverable, run the 3 Deep-Ancestry students.',
        '5. 0.6B, Router-KD, second family and diffusion are conditional breadth experiments.',
    ]
    Path(args.output).write_text('\n'.join(lines) + '\n')
    print(args.output)

if __name__ == '__main__':
    main()
