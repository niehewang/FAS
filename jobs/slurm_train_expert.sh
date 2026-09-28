#!/usr/bin/env bash
#SBATCH --job-name=fas-expert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?usage: sbatch jobs/slurm_train_expert.sh configs/experts/math.yaml}
python scripts/train_lora_expert.py --config "$CONFIG"
