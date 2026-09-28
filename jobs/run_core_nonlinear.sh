#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-generated formal nonlinear ancestry pipeline (seed-specific ancestor families).
RUN_L5=${RUN_L5:-1}
RUN_L6=${RUN_L6:-1}
RUN_L7=${RUN_L7:-1}
mkdir -p runs/{metadata,checkpoints,teacher_data_shared,deep_chain,full}

if [ "${RUN_L5}" = "1" ]; then
  echo "=== L5-mixture-kd__v2__1p7b__s11 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11'
  test -d runs/experts/s11/math/adapter || { echo "missing formal ancestor runs/experts/s11/math/adapter"; exit 2; }
  test -d runs/experts/s11/code/adapter || { echo "missing formal ancestor runs/experts/s11/code/adapter"; exit 2; }
  test -d runs/experts/s11/medical/adapter || { echo "missing formal ancestor runs/experts/s11/medical/adapter"; exit 2; }
  test -d runs/experts/s11/science/adapter || { echo "missing formal ancestor runs/experts/s11/science/adapter"; exit 2; }
  mkdir -p 'runs/teacher_data_shared/l5teacher_7dc41839cf98'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_7dc41839cf98/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_core_shared/l5teacher_7dc41839cf98.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_runs/L5-mixture-kd__v2__1p7b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__1p7b__s11' --config 'configs/generated_core_runs/L5-mixture-kd__v2__1p7b__s11/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__1p7b__s11.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s11' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter'
fi

if [ "${RUN_L5}" = "1" ]; then
  echo "=== L5-mixture-kd__v2__1p7b__s23 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23'
  test -d runs/experts/s23/math/adapter || { echo "missing formal ancestor runs/experts/s23/math/adapter"; exit 2; }
  test -d runs/experts/s23/code/adapter || { echo "missing formal ancestor runs/experts/s23/code/adapter"; exit 2; }
  test -d runs/experts/s23/medical/adapter || { echo "missing formal ancestor runs/experts/s23/medical/adapter"; exit 2; }
  test -d runs/experts/s23/science/adapter || { echo "missing formal ancestor runs/experts/s23/science/adapter"; exit 2; }
  mkdir -p 'runs/teacher_data_shared/l5teacher_76a9fbfd1e57'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_76a9fbfd1e57/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_core_shared/l5teacher_76a9fbfd1e57.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_runs/L5-mixture-kd__v2__1p7b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__1p7b__s23' --config 'configs/generated_core_runs/L5-mixture-kd__v2__1p7b__s23/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__1p7b__s23.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s23' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter'
fi

if [ "${RUN_L5}" = "1" ]; then
  echo "=== L5-mixture-kd__v2__1p7b__s47 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47'
  test -d runs/experts/s47/math/adapter || { echo "missing formal ancestor runs/experts/s47/math/adapter"; exit 2; }
  test -d runs/experts/s47/code/adapter || { echo "missing formal ancestor runs/experts/s47/code/adapter"; exit 2; }
  test -d runs/experts/s47/medical/adapter || { echo "missing formal ancestor runs/experts/s47/medical/adapter"; exit 2; }
  test -d runs/experts/s47/science/adapter || { echo "missing formal ancestor runs/experts/s47/science/adapter"; exit 2; }
  mkdir -p 'runs/teacher_data_shared/l5teacher_56e3386f8552'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_56e3386f8552/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_core_shared/l5teacher_56e3386f8552.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_runs/L5-mixture-kd__v2__1p7b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__1p7b__s47' --config 'configs/generated_core_runs/L5-mixture-kd__v2__1p7b__s47/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__1p7b__s47.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s47' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter'
fi

if [ "${RUN_L7}" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__1p7b__s11 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11'
  test -d runs/experts/s11/math/adapter || { echo "missing formal ancestor runs/experts/s11/math/adapter"; exit 2; }
  test -d runs/experts/s11/code/adapter || { echo "missing formal ancestor runs/experts/s11/code/adapter"; exit 2; }
  test -d runs/experts/s11/medical/adapter || { echo "missing formal ancestor runs/experts/s11/medical/adapter"; exit 2; }
  if [ ! -f runs/full/s11/math/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/math/adapter --output runs/full/s11/math; fi
  if [ ! -f runs/full/s11/code/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/code/adapter --output runs/full/s11/code; fi
  if [ ! -f runs/full/s11/medical/config.json ]; then mkdir -p runs/full/s11; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s11/medical/adapter --output runs/full/s11/medical; fi
  if [ ! -f 'runs/deep_chain/merge_8b4e92852479/config.json' ]; then mergekit-yaml 'configs/generated_core_shared/merge_8b4e92852479.yaml' 'runs/deep_chain/merge_8b4e92852479' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_4ee9ea11f7d6/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_shared/deepteacher_4ee9ea11f7d6_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_4ee9ea11f7d6/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_8b4e92852479' --adapter 'runs/deep_chain/deepteacher_4ee9ea11f7d6/sft/adapter' --output 'runs/deep_chain/deepteacher_4ee9ea11f7d6/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_4ee9ea11f7d6/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_core_shared/deepteacher_4ee9ea11f7d6_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_runs/L7-deep-ancestry__v1__1p7b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__1p7b__s11' --config 'configs/generated_core_runs/L7-deep-ancestry__v1__1p7b__s11/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__1p7b__s11.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s11' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter'
fi

if [ "${RUN_L7}" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__1p7b__s23 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23'
  test -d runs/experts/s23/math/adapter || { echo "missing formal ancestor runs/experts/s23/math/adapter"; exit 2; }
  test -d runs/experts/s23/code/adapter || { echo "missing formal ancestor runs/experts/s23/code/adapter"; exit 2; }
  test -d runs/experts/s23/medical/adapter || { echo "missing formal ancestor runs/experts/s23/medical/adapter"; exit 2; }
  if [ ! -f runs/full/s23/math/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/math/adapter --output runs/full/s23/math; fi
  if [ ! -f runs/full/s23/code/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/code/adapter --output runs/full/s23/code; fi
  if [ ! -f runs/full/s23/medical/config.json ]; then mkdir -p runs/full/s23; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s23/medical/adapter --output runs/full/s23/medical; fi
  if [ ! -f 'runs/deep_chain/merge_d4b7b8ced083/config.json' ]; then mergekit-yaml 'configs/generated_core_shared/merge_d4b7b8ced083.yaml' 'runs/deep_chain/merge_d4b7b8ced083' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_f0f968b86d90/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_shared/deepteacher_f0f968b86d90_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_f0f968b86d90/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_d4b7b8ced083' --adapter 'runs/deep_chain/deepteacher_f0f968b86d90/sft/adapter' --output 'runs/deep_chain/deepteacher_f0f968b86d90/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_f0f968b86d90/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_core_shared/deepteacher_f0f968b86d90_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_runs/L7-deep-ancestry__v1__1p7b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__1p7b__s23' --config 'configs/generated_core_runs/L7-deep-ancestry__v1__1p7b__s23/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__1p7b__s23.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s23' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter'
fi

if [ "${RUN_L7}" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__1p7b__s47 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47'
  test -d runs/experts/s47/math/adapter || { echo "missing formal ancestor runs/experts/s47/math/adapter"; exit 2; }
  test -d runs/experts/s47/code/adapter || { echo "missing formal ancestor runs/experts/s47/code/adapter"; exit 2; }
  test -d runs/experts/s47/medical/adapter || { echo "missing formal ancestor runs/experts/s47/medical/adapter"; exit 2; }
  if [ ! -f runs/full/s47/math/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/math/adapter --output runs/full/s47/math; fi
  if [ ! -f runs/full/s47/code/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/code/adapter --output runs/full/s47/code; fi
  if [ ! -f runs/full/s47/medical/config.json ]; then mkdir -p runs/full/s47; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/s47/medical/adapter --output runs/full/s47/medical; fi
  if [ ! -f 'runs/deep_chain/merge_9d0246654b28/config.json' ]; then mergekit-yaml 'configs/generated_core_shared/merge_9d0246654b28.yaml' 'runs/deep_chain/merge_9d0246654b28' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_4ffe995e7ec1/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_shared/deepteacher_4ffe995e7ec1_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_4ffe995e7ec1/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_9d0246654b28' --adapter 'runs/deep_chain/deepteacher_4ffe995e7ec1/sft/adapter' --output 'runs/deep_chain/deepteacher_4ffe995e7ec1/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_4ffe995e7ec1/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_core_shared/deepteacher_4ffe995e7ec1_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_core_runs/L7-deep-ancestry__v1__1p7b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__1p7b__s47' --config 'configs/generated_core_runs/L7-deep-ancestry__v1__1p7b__s47/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__1p7b__s47.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s47' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter'
fi

echo "Formal nonlinear ancestry construction finished. Audit with seed-matched banks."
