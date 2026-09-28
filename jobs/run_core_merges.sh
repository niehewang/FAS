#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/{full,checkpoints,metadata}

echo "=== L1-linear-2p__v1__same__s11 ==="
test -d runs/experts/s11/math/adapter || { echo "missing runs/experts/s11/math/adapter"; exit 2; }
if [ ! -f runs/full/s11/math/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/math/adapter --output runs/full/s11/math; fi
test -d runs/experts/s11/code/adapter || { echo "missing runs/experts/s11/code/adapter"; exit 2; }
if [ ! -f runs/full/s11/code/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/code/adapter --output runs/full/s11/code; fi
if [ ! -f runs/checkpoints/L1-linear-2p__v1__same__s11/model/config.json ]; then python scripts/merge_linear_models.py --models runs/full/s11/math runs/full/s11/code --weights 0.5 0.5 --normalize --output runs/checkpoints/L1-linear-2p__v1__same__s11/model; fi
python scripts/record_run_metadata.py --run-id 'L1-linear-2p__v1__same__s11' --config 'configs/experiment_matrix.yaml' --checkpoint 'runs/checkpoints/L1-linear-2p__v1__same__s11/model' --parents 'math;code' --output 'runs/metadata/L1-linear-2p__v1__same__s11.json'
python scripts/manifest_status.py set 'L1-linear-2p__v1__same__s11' ready --checkpoint-or-api 'runs/checkpoints/L1-linear-2p__v1__same__s11/model'

echo "=== L1-linear-2p__v1__same__s23 ==="
test -d runs/experts/s23/math/adapter || { echo "missing runs/experts/s23/math/adapter"; exit 2; }
if [ ! -f runs/full/s23/math/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/math/adapter --output runs/full/s23/math; fi
test -d runs/experts/s23/code/adapter || { echo "missing runs/experts/s23/code/adapter"; exit 2; }
if [ ! -f runs/full/s23/code/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/code/adapter --output runs/full/s23/code; fi
if [ ! -f runs/checkpoints/L1-linear-2p__v1__same__s23/model/config.json ]; then python scripts/merge_linear_models.py --models runs/full/s23/math runs/full/s23/code --weights 0.5 0.5 --normalize --output runs/checkpoints/L1-linear-2p__v1__same__s23/model; fi
python scripts/record_run_metadata.py --run-id 'L1-linear-2p__v1__same__s23' --config 'configs/experiment_matrix.yaml' --checkpoint 'runs/checkpoints/L1-linear-2p__v1__same__s23/model' --parents 'math;code' --output 'runs/metadata/L1-linear-2p__v1__same__s23.json'
python scripts/manifest_status.py set 'L1-linear-2p__v1__same__s23' ready --checkpoint-or-api 'runs/checkpoints/L1-linear-2p__v1__same__s23/model'

echo "=== L1-linear-2p__v1__same__s47 ==="
test -d runs/experts/s47/math/adapter || { echo "missing runs/experts/s47/math/adapter"; exit 2; }
if [ ! -f runs/full/s47/math/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/math/adapter --output runs/full/s47/math; fi
test -d runs/experts/s47/code/adapter || { echo "missing runs/experts/s47/code/adapter"; exit 2; }
if [ ! -f runs/full/s47/code/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/code/adapter --output runs/full/s47/code; fi
if [ ! -f runs/checkpoints/L1-linear-2p__v1__same__s47/model/config.json ]; then python scripts/merge_linear_models.py --models runs/full/s47/math runs/full/s47/code --weights 0.5 0.5 --normalize --output runs/checkpoints/L1-linear-2p__v1__same__s47/model; fi
python scripts/record_run_metadata.py --run-id 'L1-linear-2p__v1__same__s47' --config 'configs/experiment_matrix.yaml' --checkpoint 'runs/checkpoints/L1-linear-2p__v1__same__s47/model' --parents 'math;code' --output 'runs/metadata/L1-linear-2p__v1__same__s47.json'
python scripts/manifest_status.py set 'L1-linear-2p__v1__same__s47' ready --checkpoint-or-api 'runs/checkpoints/L1-linear-2p__v1__same__s47/model'

echo "=== L2-ties-3p__v1__same__s11 ==="
test -d runs/experts/s11/math/adapter || { echo "missing runs/experts/s11/math/adapter"; exit 2; }
if [ ! -f runs/full/s11/math/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/math/adapter --output runs/full/s11/math; fi
test -d runs/experts/s11/code/adapter || { echo "missing runs/experts/s11/code/adapter"; exit 2; }
if [ ! -f runs/full/s11/code/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/code/adapter --output runs/full/s11/code; fi
test -d runs/experts/s11/science/adapter || { echo "missing runs/experts/s11/science/adapter"; exit 2; }
if [ ! -f runs/full/s11/science/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/science/adapter --output runs/full/s11/science; fi
if [ ! -f runs/checkpoints/L2-ties-3p__v1__same__s11/model/config.json ]; then mergekit-yaml 'configs/generated_core_merges/L2-ties-3p__v1__same__s11.yaml' 'runs/checkpoints/L2-ties-3p__v1__same__s11/model' --cuda --lazy-unpickle; fi
python scripts/record_run_metadata.py --run-id 'L2-ties-3p__v1__same__s11' --config 'configs/generated_core_merges/L2-ties-3p__v1__same__s11.yaml' --checkpoint 'runs/checkpoints/L2-ties-3p__v1__same__s11/model' --parents 'math;code;science' --output 'runs/metadata/L2-ties-3p__v1__same__s11.json'
python scripts/manifest_status.py set 'L2-ties-3p__v1__same__s11' ready --checkpoint-or-api 'runs/checkpoints/L2-ties-3p__v1__same__s11/model'

echo "=== L2-ties-3p__v1__same__s23 ==="
test -d runs/experts/s23/math/adapter || { echo "missing runs/experts/s23/math/adapter"; exit 2; }
if [ ! -f runs/full/s23/math/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/math/adapter --output runs/full/s23/math; fi
test -d runs/experts/s23/code/adapter || { echo "missing runs/experts/s23/code/adapter"; exit 2; }
if [ ! -f runs/full/s23/code/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/code/adapter --output runs/full/s23/code; fi
test -d runs/experts/s23/science/adapter || { echo "missing runs/experts/s23/science/adapter"; exit 2; }
if [ ! -f runs/full/s23/science/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/science/adapter --output runs/full/s23/science; fi
if [ ! -f runs/checkpoints/L2-ties-3p__v1__same__s23/model/config.json ]; then mergekit-yaml 'configs/generated_core_merges/L2-ties-3p__v1__same__s23.yaml' 'runs/checkpoints/L2-ties-3p__v1__same__s23/model' --cuda --lazy-unpickle; fi
python scripts/record_run_metadata.py --run-id 'L2-ties-3p__v1__same__s23' --config 'configs/generated_core_merges/L2-ties-3p__v1__same__s23.yaml' --checkpoint 'runs/checkpoints/L2-ties-3p__v1__same__s23/model' --parents 'math;code;science' --output 'runs/metadata/L2-ties-3p__v1__same__s23.json'
python scripts/manifest_status.py set 'L2-ties-3p__v1__same__s23' ready --checkpoint-or-api 'runs/checkpoints/L2-ties-3p__v1__same__s23/model'

echo "=== L2-ties-3p__v1__same__s47 ==="
test -d runs/experts/s47/math/adapter || { echo "missing runs/experts/s47/math/adapter"; exit 2; }
if [ ! -f runs/full/s47/math/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/math/adapter --output runs/full/s47/math; fi
test -d runs/experts/s47/code/adapter || { echo "missing runs/experts/s47/code/adapter"; exit 2; }
if [ ! -f runs/full/s47/code/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/code/adapter --output runs/full/s47/code; fi
test -d runs/experts/s47/science/adapter || { echo "missing runs/experts/s47/science/adapter"; exit 2; }
if [ ! -f runs/full/s47/science/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/science/adapter --output runs/full/s47/science; fi
if [ ! -f runs/checkpoints/L2-ties-3p__v1__same__s47/model/config.json ]; then mergekit-yaml 'configs/generated_core_merges/L2-ties-3p__v1__same__s47.yaml' 'runs/checkpoints/L2-ties-3p__v1__same__s47/model' --cuda --lazy-unpickle; fi
python scripts/record_run_metadata.py --run-id 'L2-ties-3p__v1__same__s47' --config 'configs/generated_core_merges/L2-ties-3p__v1__same__s47.yaml' --checkpoint 'runs/checkpoints/L2-ties-3p__v1__same__s47/model' --parents 'math;code;science' --output 'runs/metadata/L2-ties-3p__v1__same__s47.json'
python scripts/manifest_status.py set 'L2-ties-3p__v1__same__s47' ready --checkpoint-or-api 'runs/checkpoints/L2-ties-3p__v1__same__s47/model'

echo "Core seed-aware merge descendants ready."
