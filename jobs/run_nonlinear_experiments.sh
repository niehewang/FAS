#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Auto-generated nonlinear ancestry pipeline. Review datasets/licenses and GPU paths before execution.
RUN_L5=${RUN_L5:-1}
RUN_L6=${RUN_L6:-1}
RUN_L7=${RUN_L7:-1}
mkdir -p runs/{metadata,checkpoints,teacher_data_shared,deep_chain,full}
for p in math code medical science; do test -d "runs/experts/$p/adapter" || { echo "missing runs/experts/$p/adapter"; exit 2; }; done

if [ "$RUN_L7" = "1" ]; then
# Full 4B parents are needed only by Deep Ancestry (L7).
  if [ ! -f runs/full/math/config.json ]; then python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/math/adapter --output runs/full/math; fi
  if [ ! -f runs/full/code/config.json ]; then python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/code/adapter --output runs/full/code; fi
  if [ ! -f runs/full/medical/config.json ]; then python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/medical/adapter --output runs/full/medical; fi
  if [ ! -f runs/full/science/config.json ]; then python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/science/adapter --output runs/full/science; fi
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v1__1p7b__s11 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s11'
  mkdir -p 'runs/teacher_data_shared/l5teacher_e584678624d6'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_e584678624d6/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_e584678624d6.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s11/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v1__1p7b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v1__1p7b__s11' --config 'configs/generated_runs/L5-mixture-kd__v1__1p7b__s11/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s11/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v1__1p7b__s11.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__1p7b__s11' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s11/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v1__1p7b__s23 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s23'
  mkdir -p 'runs/teacher_data_shared/l5teacher_c3d217edc501'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_c3d217edc501/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_c3d217edc501.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s23/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v1__1p7b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v1__1p7b__s23' --config 'configs/generated_runs/L5-mixture-kd__v1__1p7b__s23/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s23/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v1__1p7b__s23.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__1p7b__s23' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s23/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v1__1p7b__s47 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s47'
  mkdir -p 'runs/teacher_data_shared/l5teacher_828a58a50067'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_828a58a50067/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_828a58a50067.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s47/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v1__1p7b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v1__1p7b__s47' --config 'configs/generated_runs/L5-mixture-kd__v1__1p7b__s47/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s47/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v1__1p7b__s47.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__1p7b__s47' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v1__1p7b__s47/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v1__0p6b__s11 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s11'
  mkdir -p 'runs/teacher_data_shared/l5teacher_e584678624d6'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_e584678624d6/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_e584678624d6.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s11/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v1__0p6b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v1__0p6b__s11' --config 'configs/generated_runs/L5-mixture-kd__v1__0p6b__s11/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s11/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v1__0p6b__s11.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__0p6b__s11' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s11/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v1__0p6b__s23 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s23'
  mkdir -p 'runs/teacher_data_shared/l5teacher_c3d217edc501'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_c3d217edc501/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_c3d217edc501.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s23/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v1__0p6b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v1__0p6b__s23' --config 'configs/generated_runs/L5-mixture-kd__v1__0p6b__s23/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s23/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v1__0p6b__s23.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__0p6b__s23' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s23/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v1__0p6b__s47 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s47'
  mkdir -p 'runs/teacher_data_shared/l5teacher_828a58a50067'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_828a58a50067/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_828a58a50067.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s47/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v1__0p6b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v1__0p6b__s47' --config 'configs/generated_runs/L5-mixture-kd__v1__0p6b__s47/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s47/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v1__0p6b__s47.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v1__0p6b__s47' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v1__0p6b__s47/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v2__1p7b__s11 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11'
  mkdir -p 'runs/teacher_data_shared/l5teacher_7dc41839cf98'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_7dc41839cf98/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_7dc41839cf98.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v2__1p7b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__1p7b__s11' --config 'configs/generated_runs/L5-mixture-kd__v2__1p7b__s11/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__1p7b__s11.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s11' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s11/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v2__1p7b__s23 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23'
  mkdir -p 'runs/teacher_data_shared/l5teacher_76a9fbfd1e57'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_76a9fbfd1e57/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_76a9fbfd1e57.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v2__1p7b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__1p7b__s23' --config 'configs/generated_runs/L5-mixture-kd__v2__1p7b__s23/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__1p7b__s23.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s23' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s23/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v2__1p7b__s47 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47'
  mkdir -p 'runs/teacher_data_shared/l5teacher_56e3386f8552'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_56e3386f8552/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_56e3386f8552.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v2__1p7b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__1p7b__s47' --config 'configs/generated_runs/L5-mixture-kd__v2__1p7b__s47/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__1p7b__s47.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__1p7b__s47' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__1p7b__s47/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v2__0p6b__s11 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s11'
  mkdir -p 'runs/teacher_data_shared/l5teacher_7dc41839cf98'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_7dc41839cf98/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_7dc41839cf98.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s11/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v2__0p6b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__0p6b__s11' --config 'configs/generated_runs/L5-mixture-kd__v2__0p6b__s11/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s11/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__0p6b__s11.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__0p6b__s11' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s11/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v2__0p6b__s23 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s23'
  mkdir -p 'runs/teacher_data_shared/l5teacher_76a9fbfd1e57'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_76a9fbfd1e57/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_76a9fbfd1e57.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s23/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v2__0p6b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__0p6b__s23' --config 'configs/generated_runs/L5-mixture-kd__v2__0p6b__s23/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s23/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__0p6b__s23.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__0p6b__s23' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s23/student/adapter'
fi

if [ "$RUN_L5" = "1" ]; then
  echo "=== L5-mixture-kd__v2__0p6b__s47 ==="
  mkdir -p 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s47'
  mkdir -p 'runs/teacher_data_shared/l5teacher_56e3386f8552'
  if [ ! -s 'runs/teacher_data_shared/l5teacher_56e3386f8552/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l5teacher_56e3386f8552.yaml'; fi
  if [ ! -d 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s47/student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L5-mixture-kd__v2__0p6b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L5-mixture-kd__v2__0p6b__s47' --config 'configs/generated_runs/L5-mixture-kd__v2__0p6b__s47/student.yaml' --checkpoint 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s47/student/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L5-mixture-kd__v2__0p6b__s47.json'
python scripts/manifest_status.py set 'L5-mixture-kd__v2__0p6b__s47' ready --checkpoint-or-api 'runs/checkpoints/L5-mixture-kd__v2__0p6b__s47/student/adapter'
fi

if [ "$RUN_L6" = "1" ]; then
  echo "=== L6-router-kd-sft__v1__1p7b__s11 ==="
  mkdir -p 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11'
  mkdir -p 'runs/teacher_data_shared/l6teacher_7cb6450a32dd'
  if [ ! -s 'runs/teacher_data_shared/l6teacher_7cb6450a32dd/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l6teacher_7cb6450a32dd.yaml'; fi
  if [ ! -d 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s11/student.yaml'; fi
  if [ ! -f 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_full/config.json' ]; then python scripts/materialize_lora.py --base 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router/adapter' --output 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_full'; fi
  if [ ! -d 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s11/aux_sft.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L6-router-kd-sft__v1__1p7b__s11' --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s11/aux_sft.yaml' --checkpoint 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_sft/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L6-router-kd-sft__v1__1p7b__s11.json'
python scripts/manifest_status.py set 'L6-router-kd-sft__v1__1p7b__s11' ready --checkpoint-or-api 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s11/student_router_sft/adapter'
fi

if [ "$RUN_L6" = "1" ]; then
  echo "=== L6-router-kd-sft__v1__1p7b__s23 ==="
  mkdir -p 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23'
  mkdir -p 'runs/teacher_data_shared/l6teacher_f90eb0ce632b'
  if [ ! -s 'runs/teacher_data_shared/l6teacher_f90eb0ce632b/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l6teacher_f90eb0ce632b.yaml'; fi
  if [ ! -d 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s23/student.yaml'; fi
  if [ ! -f 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_full/config.json' ]; then python scripts/materialize_lora.py --base 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router/adapter' --output 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_full'; fi
  if [ ! -d 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s23/aux_sft.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L6-router-kd-sft__v1__1p7b__s23' --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s23/aux_sft.yaml' --checkpoint 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_sft/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L6-router-kd-sft__v1__1p7b__s23.json'
python scripts/manifest_status.py set 'L6-router-kd-sft__v1__1p7b__s23' ready --checkpoint-or-api 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s23/student_router_sft/adapter'
fi

if [ "$RUN_L6" = "1" ]; then
  echo "=== L6-router-kd-sft__v1__1p7b__s47 ==="
  mkdir -p 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47'
  mkdir -p 'runs/teacher_data_shared/l6teacher_dc3aca863adc'
  if [ ! -s 'runs/teacher_data_shared/l6teacher_dc3aca863adc/responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/l6teacher_dc3aca863adc.yaml'; fi
  if [ ! -d 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s47/student.yaml'; fi
  if [ ! -f 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_full/config.json' ]; then python scripts/materialize_lora.py --base 'Qwen/Qwen3-1.7B-Base' --adapter 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router/adapter' --output 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_full'; fi
  if [ ! -d 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s47/aux_sft.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L6-router-kd-sft__v1__1p7b__s47' --config 'configs/generated_runs/L6-router-kd-sft__v1__1p7b__s47/aux_sft.yaml' --checkpoint 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_sft/adapter' --parents 'math;code;medical;science' --output 'runs/metadata/L6-router-kd-sft__v1__1p7b__s47.json'
python scripts/manifest_status.py set 'L6-router-kd-sft__v1__1p7b__s47' ready --checkpoint-or-api 'runs/checkpoints/L6-router-kd-sft__v1__1p7b__s47/student_router_sft/adapter'
fi

if [ "$RUN_L7" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__1p7b__s11 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11'
  if [ ! -f 'runs/deep_chain/merge_52ec1cc8805e/config.json' ]; then mergekit-yaml 'configs/generated_shared/merge_52ec1cc8805e.yaml' 'runs/deep_chain/merge_52ec1cc8805e' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_c7da265705c2/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_shared/deepteacher_c7da265705c2_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_c7da265705c2/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_52ec1cc8805e' --adapter 'runs/deep_chain/deepteacher_c7da265705c2/sft/adapter' --output 'runs/deep_chain/deepteacher_c7da265705c2/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_c7da265705c2/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/deepteacher_c7da265705c2_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L7-deep-ancestry__v1__1p7b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__1p7b__s11' --config 'configs/generated_runs/L7-deep-ancestry__v1__1p7b__s11/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__1p7b__s11.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s11' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s11/deep_student/adapter'
fi

if [ "$RUN_L7" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__1p7b__s23 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23'
  if [ ! -f 'runs/deep_chain/merge_52ec1cc8805e/config.json' ]; then mergekit-yaml 'configs/generated_shared/merge_52ec1cc8805e.yaml' 'runs/deep_chain/merge_52ec1cc8805e' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_61c02313e58d/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_shared/deepteacher_61c02313e58d_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_61c02313e58d/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_52ec1cc8805e' --adapter 'runs/deep_chain/deepteacher_61c02313e58d/sft/adapter' --output 'runs/deep_chain/deepteacher_61c02313e58d/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_61c02313e58d/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/deepteacher_61c02313e58d_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L7-deep-ancestry__v1__1p7b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__1p7b__s23' --config 'configs/generated_runs/L7-deep-ancestry__v1__1p7b__s23/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__1p7b__s23.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s23' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s23/deep_student/adapter'
fi

if [ "$RUN_L7" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__1p7b__s47 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47'
  if [ ! -f 'runs/deep_chain/merge_52ec1cc8805e/config.json' ]; then mergekit-yaml 'configs/generated_shared/merge_52ec1cc8805e.yaml' 'runs/deep_chain/merge_52ec1cc8805e' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_653195cdeccc/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_shared/deepteacher_653195cdeccc_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_653195cdeccc/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_52ec1cc8805e' --adapter 'runs/deep_chain/deepteacher_653195cdeccc/sft/adapter' --output 'runs/deep_chain/deepteacher_653195cdeccc/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_653195cdeccc/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/deepteacher_653195cdeccc_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L7-deep-ancestry__v1__1p7b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__1p7b__s47' --config 'configs/generated_runs/L7-deep-ancestry__v1__1p7b__s47/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__1p7b__s47.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__1p7b__s47' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__1p7b__s47/deep_student/adapter'
fi

if [ "$RUN_L7" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__0p6b__s11 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s11'
  if [ ! -f 'runs/deep_chain/merge_52ec1cc8805e/config.json' ]; then mergekit-yaml 'configs/generated_shared/merge_52ec1cc8805e.yaml' 'runs/deep_chain/merge_52ec1cc8805e' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_c7da265705c2/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_shared/deepteacher_c7da265705c2_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_c7da265705c2/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_52ec1cc8805e' --adapter 'runs/deep_chain/deepteacher_c7da265705c2/sft/adapter' --output 'runs/deep_chain/deepteacher_c7da265705c2/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_c7da265705c2/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/deepteacher_c7da265705c2_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s11/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L7-deep-ancestry__v1__0p6b__s11/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__0p6b__s11' --config 'configs/generated_runs/L7-deep-ancestry__v1__0p6b__s11/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s11/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__0p6b__s11.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__0p6b__s11' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s11/deep_student/adapter'
fi

if [ "$RUN_L7" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__0p6b__s23 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s23'
  if [ ! -f 'runs/deep_chain/merge_52ec1cc8805e/config.json' ]; then mergekit-yaml 'configs/generated_shared/merge_52ec1cc8805e.yaml' 'runs/deep_chain/merge_52ec1cc8805e' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_61c02313e58d/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_shared/deepteacher_61c02313e58d_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_61c02313e58d/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_52ec1cc8805e' --adapter 'runs/deep_chain/deepteacher_61c02313e58d/sft/adapter' --output 'runs/deep_chain/deepteacher_61c02313e58d/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_61c02313e58d/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/deepteacher_61c02313e58d_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s23/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L7-deep-ancestry__v1__0p6b__s23/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__0p6b__s23' --config 'configs/generated_runs/L7-deep-ancestry__v1__0p6b__s23/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s23/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__0p6b__s23.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__0p6b__s23' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s23/deep_student/adapter'
fi

if [ "$RUN_L7" = "1" ]; then
  echo "=== L7-deep-ancestry__v1__0p6b__s47 ==="
  mkdir -p 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s47'
  if [ ! -f 'runs/deep_chain/merge_52ec1cc8805e/config.json' ]; then mergekit-yaml 'configs/generated_shared/merge_52ec1cc8805e.yaml' 'runs/deep_chain/merge_52ec1cc8805e' --cuda --lazy-unpickle; fi
  if [ ! -d 'runs/deep_chain/deepteacher_653195cdeccc/sft/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_shared/deepteacher_653195cdeccc_sft.yaml'; fi
  if [ ! -f 'runs/deep_chain/deepteacher_653195cdeccc/sft_full/config.json' ]; then python scripts/materialize_lora.py --base 'runs/deep_chain/merge_52ec1cc8805e' --adapter 'runs/deep_chain/deepteacher_653195cdeccc/sft/adapter' --output 'runs/deep_chain/deepteacher_653195cdeccc/sft_full'; fi
  if [ ! -s 'runs/deep_chain/deepteacher_653195cdeccc/int4_responses.jsonl' ]; then python scripts/generate_teacher_data.py --config 'configs/generated_shared/deepteacher_653195cdeccc_teacher.yaml'; fi
  if [ ! -d 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s47/deep_student/adapter' ]; then python scripts/train_lora_expert.py --config 'configs/generated_runs/L7-deep-ancestry__v1__0p6b__s47/student.yaml'; fi
python scripts/record_run_metadata.py --run-id 'L7-deep-ancestry__v1__0p6b__s47' --config 'configs/generated_runs/L7-deep-ancestry__v1__0p6b__s47/student.yaml' --checkpoint 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s47/deep_student/adapter' --parents 'math;code;medical' --output 'runs/metadata/L7-deep-ancestry__v1__0p6b__s47.json'
python scripts/manifest_status.py set 'L7-deep-ancestry__v1__0p6b__s47' ready --checkpoint-or-api 'runs/checkpoints/L7-deep-ancestry__v1__0p6b__s47/deep_student/adapter'
fi

echo "Nonlinear ancestry construction finished. Run jobs/run_nonlinear_audit.sh after the desired stages are ready."
