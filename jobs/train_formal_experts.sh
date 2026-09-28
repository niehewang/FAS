#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Formal sibling experts: independent genealogy seeds.
test -s data/training/formal/math.jsonl || { echo "missing formal corpus: data/training/formal/math.jsonl"; exit 2; }
if [ ! -d runs/experts/s11/math/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s11/math.yaml; fi
test -s data/training/formal/code.jsonl || { echo "missing formal corpus: data/training/formal/code.jsonl"; exit 2; }
if [ ! -d runs/experts/s11/code/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s11/code.yaml; fi
test -s data/training/formal/medical.jsonl || { echo "missing formal corpus: data/training/formal/medical.jsonl"; exit 2; }
if [ ! -d runs/experts/s11/medical/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s11/medical.yaml; fi
test -s data/training/formal/science.jsonl || { echo "missing formal corpus: data/training/formal/science.jsonl"; exit 2; }
if [ ! -d runs/experts/s11/science/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s11/science.yaml; fi
test -s data/training/formal/math.jsonl || { echo "missing formal corpus: data/training/formal/math.jsonl"; exit 2; }
if [ ! -d runs/experts/s23/math/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s23/math.yaml; fi
test -s data/training/formal/code.jsonl || { echo "missing formal corpus: data/training/formal/code.jsonl"; exit 2; }
if [ ! -d runs/experts/s23/code/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s23/code.yaml; fi
test -s data/training/formal/medical.jsonl || { echo "missing formal corpus: data/training/formal/medical.jsonl"; exit 2; }
if [ ! -d runs/experts/s23/medical/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s23/medical.yaml; fi
test -s data/training/formal/science.jsonl || { echo "missing formal corpus: data/training/formal/science.jsonl"; exit 2; }
if [ ! -d runs/experts/s23/science/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s23/science.yaml; fi
test -s data/training/formal/math.jsonl || { echo "missing formal corpus: data/training/formal/math.jsonl"; exit 2; }
if [ ! -d runs/experts/s47/math/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s47/math.yaml; fi
test -s data/training/formal/code.jsonl || { echo "missing formal corpus: data/training/formal/code.jsonl"; exit 2; }
if [ ! -d runs/experts/s47/code/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s47/code.yaml; fi
test -s data/training/formal/medical.jsonl || { echo "missing formal corpus: data/training/formal/medical.jsonl"; exit 2; }
if [ ! -d runs/experts/s47/medical/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s47/medical.yaml; fi
test -s data/training/formal/science.jsonl || { echo "missing formal corpus: data/training/formal/science.jsonl"; exit 2; }
if [ ! -d runs/experts/s47/science/adapter ]; then python scripts/train_lora_expert.py --config configs/formal_experts/s47/science.yaml; fi
echo "Formal seeded sibling experts ready."
