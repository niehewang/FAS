# Deep Ancestry Stress Test execution recipe

Target chain in the manuscript:

`3-parent merge -> auxiliary SFT -> INT4 inference teacher -> response KD to 1.7B/0.6B student`

Recommended reproducible implementation:

1. Materialize three sibling LoRA experts to full checkpoints.
2. Run TIES/DARE-TIES with the official mergekit and archive the exact YAML/version.
3. Copy `stage2_sft.yaml`, set `model_name` to the merged checkpoint, and train the auxiliary SFT LoRA.
4. Materialize the SFT LoRA into `runs/deep_chain/sft_full`.
5. For the INT4 stage, do **not** create an opaque custom quantized checkpoint. Load `sft_full` through bitsandbytes 4-bit (`--load-in-4bit`) for probing and teacher-response generation. Record the quantization config.
6. Run `generate_teacher_data.py --config stage4_quantized_teacher_kd.yaml` to obtain KD responses from the INT4 teacher.
7. Train Qwen3-1.7B-Base / 0.6B-Base students using the normal student config.
8. Save response banks at every intermediate stage so the manuscript can plot the Functional Ancestry Trajectory.

This implementation makes every transformation explicit and avoids depending on an untracked quantized serialization format.
