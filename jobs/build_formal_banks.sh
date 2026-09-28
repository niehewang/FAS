#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

test -s 'data/probes/interventions.jsonl' || { echo "missing formal probe pool"; exit 2; }
AUDIT_TEMPERATURE=${AUDIT_TEMPERATURE:-0.7}
AUDIT_TOP_P=${AUDIT_TOP_P:-0.95}

echo "=== FORMAL BANK s11 ==="
mkdir -p runs/banks/s11/raw runs/banks/s11/split
if [ ! -s runs/banks/s11/raw/base.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s11/raw/base.npz; fi
test -d runs/experts/s11/math/adapter || { echo "missing runs/experts/s11/math/adapter"; exit 2; }
if [ ! -s runs/banks/s11/raw/math.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s11/math/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s11/raw/math.npz; fi
test -d runs/experts/s11/code/adapter || { echo "missing runs/experts/s11/code/adapter"; exit 2; }
if [ ! -s runs/banks/s11/raw/code.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s11/code/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s11/raw/code.npz; fi
test -d runs/experts/s11/medical/adapter || { echo "missing runs/experts/s11/medical/adapter"; exit 2; }
if [ ! -s runs/banks/s11/raw/medical.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s11/medical/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s11/raw/medical.npz; fi
test -d runs/experts/s11/science/adapter || { echo "missing runs/experts/s11/science/adapter"; exit 2; }
if [ ! -s runs/banks/s11/raw/science.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s11/science/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10011 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s11/raw/science.npz; fi
python scripts/split_embedding_replicates.py runs/banks/s11/raw/base.npz --seed 20011 --select-output runs/banks/s11/split/base.select.npz --estimate-output runs/banks/s11/split/base.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s11/raw/math.npz --seed 20011 --select-output runs/banks/s11/split/math.select.npz --estimate-output runs/banks/s11/split/math.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s11/raw/code.npz --seed 20011 --select-output runs/banks/s11/split/code.select.npz --estimate-output runs/banks/s11/split/code.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s11/raw/medical.npz --seed 20011 --select-output runs/banks/s11/split/medical.select.npz --estimate-output runs/banks/s11/split/medical.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s11/raw/science.npz --seed 20011 --select-output runs/banks/s11/split/science.select.npz --estimate-output runs/banks/s11/split/science.estimate.npz
python scripts/build_response_bank.py --base runs/banks/s11/split/base.select.npz --ancestors runs/banks/s11/split/math.select.npz runs/banks/s11/split/code.select.npz runs/banks/s11/split/medical.select.npz runs/banks/s11/split/science.select.npz --names Math Code Medical Science --output runs/banks/s11/response_bank.select.npz
python scripts/build_response_bank.py --base runs/banks/s11/split/base.estimate.npz --ancestors runs/banks/s11/split/math.estimate.npz runs/banks/s11/split/code.estimate.npz runs/banks/s11/split/medical.estimate.npz runs/banks/s11/split/science.estimate.npz --names Math Code Medical Science --output runs/banks/s11/response_bank.estimate.npz
python scripts/select_probes.py runs/banks/s11/response_bank.select.npz --budget 64 --balanced --probe-jsonl data/probes/interventions.jsonl --group-key domain --output runs/banks/s11/selected_probes.json

echo "=== FORMAL BANK s23 ==="
mkdir -p runs/banks/s23/raw runs/banks/s23/split
if [ ! -s runs/banks/s23/raw/base.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10023 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s23/raw/base.npz; fi
test -d runs/experts/s23/math/adapter || { echo "missing runs/experts/s23/math/adapter"; exit 2; }
if [ ! -s runs/banks/s23/raw/math.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s23/math/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10023 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s23/raw/math.npz; fi
test -d runs/experts/s23/code/adapter || { echo "missing runs/experts/s23/code/adapter"; exit 2; }
if [ ! -s runs/banks/s23/raw/code.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s23/code/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10023 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s23/raw/code.npz; fi
test -d runs/experts/s23/medical/adapter || { echo "missing runs/experts/s23/medical/adapter"; exit 2; }
if [ ! -s runs/banks/s23/raw/medical.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s23/medical/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10023 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s23/raw/medical.npz; fi
test -d runs/experts/s23/science/adapter || { echo "missing runs/experts/s23/science/adapter"; exit 2; }
if [ ! -s runs/banks/s23/raw/science.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s23/science/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10023 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s23/raw/science.npz; fi
python scripts/split_embedding_replicates.py runs/banks/s23/raw/base.npz --seed 20023 --select-output runs/banks/s23/split/base.select.npz --estimate-output runs/banks/s23/split/base.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s23/raw/math.npz --seed 20023 --select-output runs/banks/s23/split/math.select.npz --estimate-output runs/banks/s23/split/math.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s23/raw/code.npz --seed 20023 --select-output runs/banks/s23/split/code.select.npz --estimate-output runs/banks/s23/split/code.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s23/raw/medical.npz --seed 20023 --select-output runs/banks/s23/split/medical.select.npz --estimate-output runs/banks/s23/split/medical.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s23/raw/science.npz --seed 20023 --select-output runs/banks/s23/split/science.select.npz --estimate-output runs/banks/s23/split/science.estimate.npz
python scripts/build_response_bank.py --base runs/banks/s23/split/base.select.npz --ancestors runs/banks/s23/split/math.select.npz runs/banks/s23/split/code.select.npz runs/banks/s23/split/medical.select.npz runs/banks/s23/split/science.select.npz --names Math Code Medical Science --output runs/banks/s23/response_bank.select.npz
python scripts/build_response_bank.py --base runs/banks/s23/split/base.estimate.npz --ancestors runs/banks/s23/split/math.estimate.npz runs/banks/s23/split/code.estimate.npz runs/banks/s23/split/medical.estimate.npz runs/banks/s23/split/science.estimate.npz --names Math Code Medical Science --output runs/banks/s23/response_bank.estimate.npz
python scripts/select_probes.py runs/banks/s23/response_bank.select.npz --budget 64 --balanced --probe-jsonl data/probes/interventions.jsonl --group-key domain --output runs/banks/s23/selected_probes.json

echo "=== FORMAL BANK s47 ==="
mkdir -p runs/banks/s47/raw runs/banks/s47/split
if [ ! -s runs/banks/s47/raw/base.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10047 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s47/raw/base.npz; fi
test -d runs/experts/s47/math/adapter || { echo "missing runs/experts/s47/math/adapter"; exit 2; }
if [ ! -s runs/banks/s47/raw/math.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s47/math/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10047 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s47/raw/math.npz; fi
test -d runs/experts/s47/code/adapter || { echo "missing runs/experts/s47/code/adapter"; exit 2; }
if [ ! -s runs/banks/s47/raw/code.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s47/code/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10047 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s47/raw/code.npz; fi
test -d runs/experts/s47/medical/adapter || { echo "missing runs/experts/s47/medical/adapter"; exit 2; }
if [ ! -s runs/banks/s47/raw/medical.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s47/medical/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10047 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s47/raw/medical.npz; fi
test -d runs/experts/s47/science/adapter || { echo "missing runs/experts/s47/science/adapter"; exit 2; }
if [ ! -s runs/banks/s47/raw/science.npz ]; then python scripts/run_text_probe_bank.py --model Qwen/Qwen3-4B-Base --adapter runs/experts/s47/science/adapter --probes 'data/probes/interventions.jsonl' --encoder 'sentence-transformers/all-mpnet-base-v2' --samples 4 --seed 10047 --temperature "$AUDIT_TEMPERATURE" --top-p "$AUDIT_TOP_P" --paired-seeds --output runs/banks/s47/raw/science.npz; fi
python scripts/split_embedding_replicates.py runs/banks/s47/raw/base.npz --seed 20047 --select-output runs/banks/s47/split/base.select.npz --estimate-output runs/banks/s47/split/base.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s47/raw/math.npz --seed 20047 --select-output runs/banks/s47/split/math.select.npz --estimate-output runs/banks/s47/split/math.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s47/raw/code.npz --seed 20047 --select-output runs/banks/s47/split/code.select.npz --estimate-output runs/banks/s47/split/code.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s47/raw/medical.npz --seed 20047 --select-output runs/banks/s47/split/medical.select.npz --estimate-output runs/banks/s47/split/medical.estimate.npz
python scripts/split_embedding_replicates.py runs/banks/s47/raw/science.npz --seed 20047 --select-output runs/banks/s47/split/science.select.npz --estimate-output runs/banks/s47/split/science.estimate.npz
python scripts/build_response_bank.py --base runs/banks/s47/split/base.select.npz --ancestors runs/banks/s47/split/math.select.npz runs/banks/s47/split/code.select.npz runs/banks/s47/split/medical.select.npz runs/banks/s47/split/science.select.npz --names Math Code Medical Science --output runs/banks/s47/response_bank.select.npz
python scripts/build_response_bank.py --base runs/banks/s47/split/base.estimate.npz --ancestors runs/banks/s47/split/math.estimate.npz runs/banks/s47/split/code.estimate.npz runs/banks/s47/split/medical.estimate.npz runs/banks/s47/split/science.estimate.npz --names Math Code Medical Science --output runs/banks/s47/response_bank.estimate.npz
python scripts/select_probes.py runs/banks/s47/response_bank.select.npz --budget 64 --balanced --probe-jsonl data/probes/interventions.jsonl --group-key domain --output runs/banks/s47/selected_probes.json

echo "All formal seed-specific ancestry banks ready."
