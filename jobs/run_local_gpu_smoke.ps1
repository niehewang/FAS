$ErrorActionPreference = "Stop"
Write-Host "=== FAS local GPU smoke ==="
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see CUDA'; print(torch.cuda.get_device_name(0))"
python -m pip install -q -U "transformers>=4.51" "datasets>=3.0" "peft>=0.15" "accelerate>=1.2" "bitsandbytes>=0.45" "sentence-transformers>=3.3" scipy scikit-learn
python scripts/hf_jobs/qwen_clean_merge_job.py --smoke --seed 11 --base-model Qwen/Qwen3-4B-Base
