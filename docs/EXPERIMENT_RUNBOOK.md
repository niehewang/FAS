# v0.9 正式 staged execution（优先使用）

v0.9 把“完整设计空间”和“默认昂贵执行”分开。`configs/experiment_matrix.yaml` 仍展开为 54 个逻辑 runs，但 `configs/evidence_tiers.yaml` 只将 15 个 row 设为 core。正式 seed 11/23/47 均对应独立训练的 sibling ancestor quartet；不能把同一套 experts 的不同 decoding seed 当作三次 genealogy 重复。

```bash
make formal-plan
make execution-tiers
```

默认正式流程：
```bash
bash jobs/train_formal_experts.sh       # 12 个 LoRA：4 domains x 3 genealogy seeds
bash jobs/build_formal_banks.sh         # 每个 seed 独立 cross-fit bank + active probes
bash jobs/run_core_merges.sh            # L1/L2
bash jobs/run_g0_calibration_s11.sh
bash jobs/run_g0_calibration_s23.sh
bash jobs/run_g0_calibration_s47.sh
bash jobs/run_core_nonlinear.sh         # L5/L7 仅 1.7B core
bash jobs/run_core_formal_audit.sh
bash jobs/run_core_open_set.sh          # C5 主结果只需 s11；s23/s47 为条件复制
```

默认 core=15：L1-v1 x3、L2-v1 x3、L5-v2-1.7B x3、L7-v1-1.7B x3、L8 三类 x1 seed。其他 39 rows 都是 conditional。完整 GPU stop/go 和 artifact 复用见 `EXECUTION_BUDGET.md`。G0 finite-sample conformal claim 只属于真正 matched 的 weighted-adapter composition family；L1/L2 weight merge、KD、Deep Chain 默认只使用 `g0_transfer/g1_transfer` 诊断，除非专门构造 matched calibration。

## Open-set 的低成本主设计
seed11 主实验把 verifier bank 限定为 Math/Code/Medical。`Unknown-A` 的 descendant 由 Math+Code+Science 构造，其中 Science 对实验者已知、对 verifier 候选库不可见；因此不需要再训练第五个“未知”expert。`Unknown-B` 直接复用 s11 Deep-Chain known-parent descendant，检验重变换是否被错误解释成 unseen ancestry。Known-only / Unknown-A 使用 restricted-bank matched adapter-composition calibration；Unknown-B 只作为 G1 hard negative。

## Optional diffusion breadth
`run_diffusion_probe_bank.py` 已输出与文本 runner 完全相同的 `base_embeddings / edited_embeddings / probe_ids / global_probe_indices` NPZ schema，因此下游 response bank、active selection、FAS decomposition 全部复用。Diffusion 属于 C7 breadth validation，默认不在 core GPU 计划中。

# FAS / AncestryBench 实验执行手册

目标：尽可能把后续作者工作压缩为 **训练/推理 -> 保存原始预测 -> 自动计算指标 -> 自动生成论文资产**。

## 0. 先验证代码环境
```bash
pip install -r requirements.txt
make test
make sanity
```
`sanity_demo.py` 不下载任何模型，只验证 log-det selection、NNLS decomposition 和核心代码路径。

生成统一实验 run manifest：
```bash
python scripts/expand_experiment_matrix.py
```
之后所有昂贵运行均使用 `data/metadata/experiment_run_manifest.csv` 中的稳定 `run_id` 命名，避免后期无法追踪某张表来自哪个 checkpoint。


## 0.5 Pilot 生死实验：从空目录到四个 sibling experts
第一轮建议直接使用项目内置 pilot 数据脚本验证机制，不把时间先花在大型语料整理上：
```bash
pip install -r requirements-experiments.txt
python scripts/prepare_pilot_datasets.py --output-dir data/training/pilot
```
工程已经预生成 `configs/experts/math.yaml`、`code.yaml`、`medical.yaml`、`science.yaml`，默认指向 pilot JSONL，其他训练参数完全相同。直接依次运行：
```bash
python scripts/train_lora_expert.py --config configs/experts/math.yaml
python scripts/train_lora_expert.py --config configs/experts/code.yaml
python scripts/train_lora_expert.py --config configs/experts/medical.yaml
python scripts/train_lora_expert.py --config configs/experts/science.yaml
```
这些 pilot corpus 只用于验证 IFRF/active-probe 假设，最终 TPAMI 主实验需要更大且完成许可/污染审查的训练集。


### 推荐：先做 exact functional-mixture death test

`make pilot-plan` 生成的 `jobs/run_pilot.sh` 默认不会实体化四份完整 4B expert checkpoint。它直接用 `base + LoRA adapter` 采集祖先 responses，在独立 `bank-estimate` 上构造严格满足 $y=\sum_i w_i S_i$ 的功能混合目标，并按 $\pi_i^{ref}\propto w_i c_i$ 自动生成参考坐标。这是 active probing / identifiability 的第一生死实验。只有当该机制实验通过后，再设置：

```bash
RUN_REAL_MERGES=1 bash jobs/run_pilot.sh
```

来实体化完整 experts 并构造真实 weight merges。真实 merge weights 不被当作 FAS coordinate ground truth。

### 构造第一批 descendants
若需要显式 full checkpoints，先：
```bash
python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter runs/experts/math/adapter --output runs/full/math
# 对 Code/Medical/Science 重复
```
受控 linear merge：
```bash
python scripts/merge_linear_models.py \
  --models runs/full/math runs/full/code \
  --weights 0.5 0.5 --normalize \
  --output runs/descendants/math_code_5050
```
TIES/DARE-TIES 不在本仓库重写，按 `configs/merges/README.md` 调用正式 mergekit 并保存实际 YAML/version。

### 构造 multi-teacher student
准备一个与专家训练数据分离的 prompt JSONL，然后：
```bash
python scripts/generate_teacher_data.py --config configs/kd/mixture_kd.yaml
python scripts/train_lora_expert.py --config configs/kd/student_1p7b.yaml
```
Router-KD 则使用 `configs/kd/router_kd.yaml`。0.6B 跨尺度学生已有 `student_0p6b.yaml`。

### 真实 probe response bank
把正式 intervention pool 存为 `data/probes/interventions.jsonl`，先检查：
```bash
python scripts/validate_probe_pool.py data/probes/interventions.jsonl
```
然后对 base/每个 ancestor 各运行一次：
```bash
python scripts/run_text_probe_bank.py \
  --model Qwen/Qwen3-4B-Base \
  --probes data/probes/interventions.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --samples 4 --paired-seeds \
  --output runs/responses/base.npz

python scripts/run_text_probe_bank.py \
  --model Qwen/Qwen3-4B-Base --adapter runs/experts/math/adapter \
  --probes data/probes/interventions.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --samples 4 --paired-seeds \
  --output runs/responses/math.npz
```
对其余 ancestors 重复，随后：
```bash
python scripts/build_response_bank.py \
  --base runs/responses/base.npz \
  --ancestors runs/responses/math.npz runs/responses/code.npz runs/responses/medical.npz runs/responses/science.npz \
  --names Math Code Medical Science \
  --output runs/response_bank.npz
python scripts/select_probes.py runs/response_bank.npz --budget 64 --output runs/selected_probes.json
```

### LLM DNA baseline
```bash
pip install -r requirements-baselines.txt
python scripts/run_llm_dna_baseline.py --models <ancestor/full-model paths...> --names Math Code Medical Science --output runs/dna_ancestors.npz
```
目标 DNA 提取后用 `scripts/decompose_vectors.py` 做完全相同的 NNLS 分解。这样 DNA-Decomp 不会因为求解器不同而被弱化。

## 1. 第一阶段（必须先做）
1. 从同一个 base 得到 Math / Code / Medical / Science 四个 sibling experts。
2. 填 `data/metadata/lineage_manifest_template.csv` 并另存为正式 lineage manifest。
3. 将 `data/probes/interventions_template.jsonl` 扩展到 >=1000 条 probes。
4. 对 base + 4 ancestors 运行所有 probes，保存输出和 encoder vectors。
5. 整理为 `task_fields.npy/npz`，形状 `[N_probes, K_ancestors, d]`。
6. 运行：
```bash
python scripts/select_probes.py response_bank.npz --budget 64 --output selected_probes.json
```
7. 构造 2-parent / 3-parent controlled descendants，比较 random vs active。
8. 将汇总值填到 `results/templates/active_probe_curve.csv`。

### 第一阶段停止条件
如果 active probes 不能稳定改善字典 conditioning，或者 conditioning 与 ancestry error 无明显关系，不要立即上多教师蒸馏；先检查 intervention 质量、输出 encoder、common-base cancellation 和 response normalization。

## 2. 主实验顺序
按成本从低到高：
1. Clean weight merges
2. Merge + SFT
3. Merge + quantization
4. Mixture-KD
5. Router-KD
6. Deep chain: Merge -> SFT -> INT4 -> KD
7. Open-set Unknown-A / Unknown-B
8. 第二 LLM family
9. SDXL LoRA cross-modality

## 3. 结果文件与自动评估
优先把原始输出保存成 `results/raw_templates/` 所示格式，然后运行指标脚本，而不是人工计算：
```bash
python scripts/evaluate_decompositions.py results/raw/decomposition_predictions.csv
python scripts/compute_shapley.py results/raw/subset_utilities.csv
python scripts/compute_functional_validity.py results/raw/functional_vectors.csv
python scripts/compute_open_set_metrics.py results/raw/open_set_scores.csv
python scripts/compute_geometry_mechanism.py results/raw/geometry_runs.csv
python scripts/sync_paper_results.py
python scripts/build_paper_assets.py
```

论文层面仍使用这些模板 CSV：
- `main_results.csv`
- `scaling.csv`
- `functional_validity.csv`
- `open_set.csv`
- `crossscale.csv`
- `crossmodality.csv`
- `ablation.csv`
- `active_probe_curve.csv`
- `ancestry_trajectory.csv`
- `ancestry_profile.csv`
- `sampling_budget.csv`
- `geometry_mechanism.csv`
- `encoder_sensitivity.csv`
- `probe_type_stats.csv`

能够从原始结果唯一计算的项由脚本同步；只有无法唯一自动推断的高级汇总才需要手工填。

填写/同步后运行 `python scripts/build_paper_assets.py` 会更新表、图和保守的自动结果文字。

## 4. 不要手工做的事情
- 不要手工把数字复制进 LaTeX 表；填 CSV。
- 不要手工重画 active-probe curve / ancestry profile / trajectory；脚本生成。
- 不要把 residual 直接命名为 unknown-parent percentage。
- 不要把 FAS coordinate 直接称为 copyright percentage / causal contribution。
- 不要在 test descendants 上调 support threshold。

## 5. 论文最终还需要作者人工确认的少量内容
1. 作者/单位/基金信息；
2. 最终训练数据及许可证；
3. 最终模型 checkpoint ID/hash；
4. GPU/训练预算；
5. 实验失败案例是否支持论文现有机制解释；
6. 英文翻译阶段的表述与 TPAMI 最新模板核对。

## v0.4：三种 calibration split（禁止复用）

在正式实验中固定：

1. `cal-signal`：base repeats / base-like controls，仅用于 Low-Signal gate；
2. `cal-support`：bank-complete known descendants，仅用于选择 `tau_pi`；
3. `cal-open`：由与主 G0 保证实际一致的 bank-complete weighted-adapter composition generator 产生，仅用于 residual conformal test；每个候选 calibration descendant 必须先用固定、独立的 `cal-signal` 通过同一 Low-Signal gate，只有 signal-pass residual 才进入 conformal 零分布；task-vector/SFT/KD/deep-chain 默认属于 G1 shift，不冒充 matched G0；
4. `test`：最终报告，不能反向选择任何阈值。

配置见 `configs/calibration_protocol.yaml`。如果资源不足必须复用 calibration descendants，则论文中的有限样本保证需要降级为经验性说明。

## v0.4：每批昂贵运行后的固定动作

```bash
python scripts/record_run_metadata.py \
  --run-id <RUN_ID> --config <CONFIG> --checkpoint <CHECKPOINT> \
  --parents 'math;code' --output runs/metadata/<RUN_ID>.json
python scripts/manifest_status.py set <RUN_ID> done --checkpoint-or-api <CHECKPOINT>
```

每周至少运行一次：
```bash
python scripts/manifest_status.py summary
python scripts/preflight_submission.py
```

## v0.4：负对照顺序

在任何 headline experiment 之前先完成：
- base/base-quantized Low-Signal false-positive test；
- genealogy-label permutation chance F1；
- within-domain probe-pair shuffle。

如果 pair shuffle 仍能保持高恢复率，优先排查输出格式、domain lexical cue、encoder shortcut，而不是继续扩大模型规模。


## v0.4：选择性分解的标准命令
准备 target/calibration 的 decomposition NPZ 后，不要直接读取 raw NNLS 坐标作为论文结果。先构造互斥的 calibration scores：
```bash
python scripts/collect_calibration_scores.py signal 'runs/cal_signal/*.npz' --output runs/calibration/signal_scores.csv
python scripts/collect_calibration_scores.py open 'runs/cal_open/*.npz' --signal-calibration runs/calibration/signal_scores.csv --alpha-signal 0.05 --output runs/calibration/open_scores.csv
python scripts/check_calibration_resolution.py
```
再对测试目标执行：
```bash
python scripts/decompose_target.py runs/targets/TARGET.npz \
  --threshold <CAL_SUPPORT_TAU> \
  --signal-calibration runs/calibration/signal_scores.csv \
  --open-calibration runs/calibration/open_scores.csv \
  --alpha-signal 0.05 --alpha-open 0.05 \
  --output runs/targets/TARGET.decomposition.json
```
只有 `Decomposable` 状态进入 Parent-F1 / coordinate-L1 的闭集数值汇总；Low-Signal 与 Bank-Insufficient 单独作为 selective outcome 统计。Open-set AUROC/FPR95 默认条件化在 signal gate 已通过的目标上。

## conformal 分辨率与主阈值
`+1` rank p-value 的最小值是 `1/(n+1)`。主文固定 `alpha=0.05`：默认 `cal-signal=40`；`cal-open` 从约 50 个 G0 bank-complete 候选开始，经固定 signal gate 后目标保留 40 个 residual scores；`cal-support=24`。`alpha=0.01` 只有在真正获得至少 100 个可交换 scores 时才作为补充结果。重型 KD/deep-chain 默认视为 G1 calibration-shift stress tests，除非专门建立 matched calibration set，否则其 p-value 只作 transfer diagnostic，不能声称有限样本保证。

## v0.4：ancestor-bank cross-fitting
随机生成模型至少保留 2 个（建议正式实验 >=4）独立 replicate，并在任何 active selection 之前执行：
```bash
python scripts/split_embedding_replicates.py ancestor.npz \
  --select-output ancestor.select.npz \
  --estimate-output ancestor.estimate.npz
```
所有 active probe selection 只读取 `.select.npz` 构建的 response bank；最终字典、几何报告和 target decomposition 只读取 `.estimate.npz` 构建的 bank。禁止把 estimate split 反向用于挑选 probes。


## 校准资源自动规划
先运行：
```bash
python scripts/generate_calibration_plan.py
```
它会生成 role-disjoint 的 signal/support/open/G1 manifests。主 matched conformal 协议使用 0.05；G1 中的 KD/deep-chain 结果单独报告 calibration shift，不与 G0 finite-sample guarantee 混写。


## v0.7：随机审计与非线性流水线
- 当 `m>1` 时，`run_text_probe_bank.py` 强制 `temperature>0`；当前预注册默认是 `temperature=0.7, top_p=0.95, m=4`。温度为 0 时必须使用 `m=1`，否则重复样本完全相同，bank cross-fitting 没有意义。
- 在线 suspect 审计只 materialize selected $B$ probes；`global_probe_indices` 被写入 NPZ，并在 target/base/selection 三方不一致时直接报错。
- `RUN_L5/RUN_L6/RUN_L7` 可分别控制 Mixture-KD、Router-KD+SFT、Deep Chain。teacher responses 与 Deep Chain teacher stages 尽可能跨 student size 复用。
- L5/L6/L7 默认 `CALIBRATION_REGIME=g1_transfer`。如果没有真正 matched 的 calibration descendants，不得把 transfer p-value 描述为 conformal finite-sample guarantee。


## v0.9：从模型到论文数字的闭环

### G0 matched calibration
```bash
python scripts/generate_calibration_plan.py
python scripts/generate_calibration_commands.py
```
`jobs/run_g0_calibration.sh` 自动构造 `cal-signal / cal-support / cal-open`、selected-only 审计、score 汇总和 support threshold。 结束前会以实际生成的 `signal_scores.csv/open_scores.csv` 重新检查 conformal 分辨率和 retained count；不足计划数时必须扩充 candidate pool 后重跑。当前 G0 的 finite-sample 声明严格限定于执行器真实构造的 **weighted-adapter composition** family；SFT、task-vector、KD、Deep Chain 未建立 matched generator 时只作为 G1 shift。

### 三父 exact Shapley + domain-conditioned FAS
```bash
python scripts/generate_shapley_commands.py
bash jobs/run_shapley_adapter_merge.sh
```
默认父模型为 Math/Medical/Science，生成全部 $2^3$ subsets。默认 `--subset-weight-rule fixed`：各 subset 保留 full descendant 中已存在 parent 的原始系数，缺失 parent 系数置零；`renormalized` 只作为敏感性实验。每个功能域 $\mathcal D$ 都从该域 intervention 子池独立做 active selection，并仅用同域 probes 估计 $\pi^F(\mathcal D)$ 后与 $\varphi(\mathcal D)$ 比较。禁止将一个 global FAS vector 与不同域 Shapley 相关。Pilot utility 不执行生成代码；Code 域仍参与 ancestry recovery，但不进入这套低成本 exact-Shapley gate。

### 强基线
```bash
pip install -r requirements-baselines.txt
python scripts/generate_baseline_target_manifest.py
python scripts/generate_baseline_commands.py --checkpoint-map configs/mergekit/checkpoint_map_template.yaml
bash jobs/run_baselines.sh
```
LLM-DNA 使用上游包生成 DNA，再使用与 FAS 一致的 NNLS decomposition；modelDNA 直接调用上游 CLI，只在其开放权重/同形状 merge 假设适用的 setting 运行。job 最后自动运行 `evaluate_decompositions.py -> sync_paper_results.py -> build_paper_assets.py`。

## v0.10：正式训练数据与 probe pool（正式 GPU 实验前必须完成）

正式语料固定为四个 source-disjoint domain corpora：OpenMathInstruct-2 / Code-Feedback / MedMCQA / STEM science-text。审计 seed pool 来自 MATH test / HumanEval+MBPP+ / PubMedQA / ARC，与上述训练 repositories 不复用。配置位于 `configs/data/formal_sources.yaml` 与 `configs/data/probe_sources.yaml`。

```bash
make data-protocol
make formal-data-stage1
```

数据准备先采样 3,600 条/域，冻结审计 seeds 后做 exact + 8-word n-gram contamination screening，最终保持恰好 3,000 条/域。不要在污染过滤后用不同域不同数量继续训练。随后人工审核 `data/governance/probe_review_sample.tsv`（20 条/域）并执行：

```bash
make formal-data-freeze
```

生成的 `formal_training_manifest.json`、`formal_data_lock.json`、`contamination_report.json`、probe review TSV 与最终 intervention JSONL 均属于正式实验 provenance，应随运行归档。三个 genealogy seeds 共用同一 canonical 3,000-example/domain corpus，但独立训练 LoRA、数据 shuffle 和 ancestry bank；这样控制 domain content 同时避免伪重复。
