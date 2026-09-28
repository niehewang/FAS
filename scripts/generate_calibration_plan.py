#!/usr/bin/env python3
"""Generate a deterministic, role-disjoint calibration plan for FAS.

This script creates metadata only; it does not train or merge models.  The goal
is to remove manual bookkeeping and make the conformal protocol auditable.

Roles:
  cal-signal : base-like/no-task-delta audit episodes or matched controls.
  cal-support: known bank-complete compositions used only to tune tau_pi.
  cal-open   : bank-complete descendants sampled from matched G0.  We generate
               a candidate pool larger than the desired retained score count,
               because low-signal candidates are filtered before residual
               calibration.
  g1-shift   : optional heavy transformations used only as calibration-shift
               stress tests unless a separate matched calibration set is built.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def choose_mix(rng: np.random.Generator, parents: list[str], min_k=2, max_k=None):
    max_k = max_k or len(parents)
    k = int(rng.integers(min_k, max_k + 1))
    chosen = list(rng.choice(parents, size=k, replace=False))
    # alpha > 1 avoids pathological almost-single-parent mixtures in calibration.
    w = rng.dirichlet(np.full(k, 1.5))
    return chosen, {p: float(v) for p, v in zip(chosen, w)}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/calibration_protocol.yaml')
    ap.add_argument('--parents', default='Math,Code,Medical,Science')
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--output-dir', default='runs/calibration/plan')
    ap.add_argument('--g1-shift-n', type=int, default=12)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    parents = [x.strip() for x in args.parents.split(',') if x.strip()]
    rng = np.random.default_rng(args.seed)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    n_sig = int(cfg['splits']['cal_signal']['planned_n'])
    n_sup = int(cfg['splits']['cal_support']['planned_n'])
    n_open_ret = int(cfg['splits']['cal_open']['planned_n'])
    n_open_cand = int(cfg['splits']['cal_open'].get('candidate_pool_n', max(n_open_ret, int(np.ceil(1.25*n_open_ret)))))

    sig_fams = ['bf16_base_episode','int8_base_episode','int4_base_episode']
    signal=[]
    for i in range(n_sig):
        fam=sig_fams[i % len(sig_fams)]
        signal.append({
            'calibration_id': f'csig_{i:03d}', 'role':'cal_signal', 'family':fam,
            'seed': int(rng.integers(0, 2**31-1)), 'expected_state':'low_signal',
            'may_share_model': bool('episode' in fam),
            'note':'audit episode is an independent stochastic/query realization; use transformation-matched controls when deployment noise floor differs',
        })

    support=[]
    for i in range(n_sup):
        subset,weights=choose_mix(rng,parents,2,min(4,len(parents)))
        support.append({
            'calibration_id':f'csup_{i:03d}','role':'cal_support','family':'known_composition',
            'seed':int(rng.integers(0,2**31-1)), 'parents':json.dumps(subset),
            'weights':json.dumps(weights,sort_keys=True), 'expected_state':'decomposable',
        })

    # G0 finite-sample calibration must match what the executable constructor
    # actually builds.  v0.9 therefore uses only weighted adapter composition
    # here; task-vector/SFT/KD/deep-chain transformations remain G1 shift tests
    # unless a dedicated matched generator is implemented.
    light_fams=['adapter_merge']
    open_rows=[]
    for i in range(n_open_cand):
        subset,weights=choose_mix(rng,parents,2,min(4,len(parents)))
        fam=light_fams[i % len(light_fams)]
        open_rows.append({
            'calibration_id':f'copen_{i:03d}','role':'cal_open_candidate','generator':'G0',
            'family':fam,'seed':int(rng.integers(0,2**31-1)),
            'parents':json.dumps(subset),'weights':json.dumps(weights,sort_keys=True),
            'expected_bank_complete':True,'signal_filter_required':True,
            'target_retained_total':n_open_ret,
        })

    g1=[]
    heavy=['merge_quant_kd','deep_chain','router_kd_sft']
    for i in range(args.g1_shift_n):
        subset,weights=choose_mix(rng,parents,2,min(4,len(parents)))
        g1.append({
            'calibration_id':f'g1_{i:03d}','role':'shift_stress_test','generator':'G1',
            'family':heavy[i % len(heavy)],'seed':int(rng.integers(0,2**31-1)),
            'parents':json.dumps(subset),'weights':json.dumps(weights,sort_keys=True),
            'finite_sample_claim':'none_unless_separately_matched_calibrated',
        })

    write_csv(out/'cal_signal_plan.csv',signal)
    write_csv(out/'cal_support_plan.csv',support)
    write_csv(out/'cal_open_candidates.csv',open_rows)
    write_csv(out/'g1_shift_plan.csv',g1)

    manifest={
        'seed':args.seed,'parents':parents,
        'counts':{'cal_signal':len(signal),'cal_support':len(support),'cal_open_candidates':len(open_rows),'cal_open_target_retained':n_open_ret,'g1_shift':len(g1)},
        'alpha_protocol':cfg['alphas'],
        'role_disjoint_by_construction':True,
        'notes':cfg.get('notes',{}),
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output_dir':str(out),'counts':manifest['counts']},ensure_ascii=False))

if __name__=='__main__':
    main()
