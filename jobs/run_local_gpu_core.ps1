$ErrorActionPreference = "Stop"
Write-Host "=== FAS local GPU core: Qwen3-4B siblings + clean merge + 1.7B Mixture-KD ==="
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see CUDA'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/2**30,1))"
python -m pip install -q -U "transformers>=4.51" "datasets>=3.0" "peft>=0.15" "accelerate>=1.2" "bitsandbytes>=0.45" "sentence-transformers>=3.3" scipy
New-Item -ItemType Directory -Force -Path "runs/local_gpu_logs" | Out-Null
foreach ($seed in 11,23,47) {
  Write-Host "=== genealogy seed $seed ==="
  python scripts/hf_jobs/qwen_clean_merge_job.py --stage clean_kd --seed $seed --base-model Qwen/Qwen3-4B-Base --student-model Qwen/Qwen3-1.7B-Base --train-n 3000 --probe-per-domain 64 --budget 64 --distill-n-each 160 2>&1 | Tee-Object -FilePath "runs/local_gpu_logs/seed_$seed.log"
}
Write-Host "All requested seeds finished. Extract lines beginning with FAS_RESULT_JSON=."
