#!/usr/bin/env bash
#SBATCH --job-name=fas-probes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?usage: sbatch jobs/slurm_probe_bank.sh MODEL PROBES OUTPUT [ADAPTER]}
PROBES=${2:?usage: sbatch jobs/slurm_probe_bank.sh MODEL PROBES OUTPUT [ADAPTER]}
OUTPUT=${3:?usage: sbatch jobs/slurm_probe_bank.sh MODEL PROBES OUTPUT [ADAPTER]}
ADAPTER=${4:-}
ENCODER=${ENCODER:-sentence-transformers/all-mpnet-base-v2}
SAMPLES=${SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
ARGS=(--model "$MODEL" --probes "$PROBES" --encoder "$ENCODER" --samples "$SAMPLES" --temperature "$TEMPERATURE" --top-p "$TOP_P" --paired-seeds --output "$OUTPUT")
if [ -n "$ADAPTER" ]; then ARGS+=(--adapter "$ADAPTER"); fi
python scripts/run_text_probe_bank.py "${ARGS[@]}"
