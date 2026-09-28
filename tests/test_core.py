import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from fas_core.decompose import fas_decompose
from fas_core.conformal import conformal_pvalue
from fas_core.active_probe import greedy_logdet_select, dictionary_geometry


def test_exact_nnls():
    A=np.eye(3)
    y=np.array([0.7,0.0,0.3])
    out=fas_decompose(A,y)
    assert np.allclose(out["pi"],[0.7,0.0,0.3],atol=1e-7)
    assert out["support"]==[0,2]


def test_conformal_bounds():
    p=conformal_pvalue([0.1,0.2,0.3,0.4],0.35)
    assert 0 < p <= 1


def test_greedy_unique():
    blocks=[np.array([[1.,0.]]),np.array([[0.,1.]]),np.array([[1.,1.]])]
    sel,_=greedy_logdet_select(blocks,2,ridge=1e-3)
    assert len(sel)==2 and len(set(sel))==2


def test_dictionary_geometry_raw_and_unit():
    A=np.array([[2.,0.],[0.,1.],[0.,0.]])
    g=dictionary_geometry(A)
    assert np.isclose(g["sigma_min_raw"],1.0)
    assert np.isclose(g["sigma_min_unit"],1.0)
    assert np.isclose(g["coherence"],0.0)
    assert np.isclose(g["condition_number_raw"],2.0)
from fas_core.selective import signal_pvalue, selective_decision
from fas_core.bootstrap import bootstrap_fas


def test_signal_gate_and_selective_states():
    p = signal_pvalue([0.1, 0.2, 0.3, 0.4], 1.0)
    assert p == 0.2
    assert selective_decision(0.2, alpha_signal=0.05)["state"] == "low_signal"
    assert selective_decision(0.01, 0.01, alpha_signal=0.05, alpha_open=0.05)["state"] == "bank_insufficient"
    assert selective_decision(0.01, 0.2, alpha_signal=0.05, alpha_open=0.05)["state"] == "decomposable"


def test_bootstrap_fas_shapes_and_stability():
    A = np.eye(2)
    # B=2 probes, m=5 replicates, d=1. Means are near [0.75,0.25].
    X = np.array([
        [[0.72],[0.75],[0.78],[0.74],[0.76]],
        [[0.23],[0.25],[0.27],[0.24],[0.26]],
    ])
    out = bootstrap_fas(A, X, n_boot=100, support_threshold=0.05, seed=1)
    assert out["pi_mean"].shape == (2,)
    assert np.all(out["pi_ci_low"] <= out["pi_ci_high"])
    assert out["support_frequency"].min() > 0.9


def test_weighted_decomposition_uses_one_geometry():
    A = np.eye(2)
    y = np.array([0.8, 0.2])
    w = np.array([2.0, 0.5])
    out = fas_decompose(A, y, weights=w)
    # An exactly representable target remains exactly representable after a
    # fixed invertible diagonal whitening transform.
    assert np.allclose(out["pi"], [0.8, 0.2], atol=1e-8)
    assert np.isclose(out["rho"], 1.0, atol=1e-10)
    assert np.isclose(out["relative_residual"], 0.0, atol=1e-10)


def test_signal_score_respects_whitening():
    from fas_core.selective import signal_score
    y = np.array([3.0, 4.0])
    assert np.isclose(signal_score(y), 5.0)
    assert np.isclose(signal_score(y, weights=np.array([2.0, 0.5])), np.sqrt(40.0))


def test_functional_mixture_reference_respects_global_scaling():
    from fas_core.response import functional_mixture_reference, stack_selected
    fields=np.zeros((2,2,1),float)
    fields[:,0,0]=[3.0,4.0]  # global norm 5
    fields[:,1,0]=[0.0,2.0]  # global norm 2
    w=np.array([0.5,0.5])
    y,gamma,pi,norms=functional_mixture_reference(fields,w)
    assert np.allclose(norms,[5.0,2.0])
    assert np.allclose(gamma,[2.5,1.0])
    assert np.allclose(pi,[2.5/3.5,1.0/3.5])
    A=stack_selected(fields,[0,1],norms)
    dec=fas_decompose(A,y.reshape(-1))
    assert np.allclose(dec["pi"],pi,atol=1e-8)


def test_pilot_job_starts_from_repo_root(tmp_path):
    import subprocess
    out = tmp_path / "run_pilot.sh"
    subprocess.run([sys.executable, str(ROOT/"scripts/generate_pilot_commands.py"), "--output", str(out)], cwd=ROOT, check=True, capture_output=True, text=True)
    text = out.read_text(encoding="utf-8")
    assert 'cd "$(dirname "$0")/.."' in text
    assert 'cd "$(dirname "$0")/../.."' not in text


def test_prepare_target_selected_only(tmp_path):
    import subprocess, json
    # Global bank has 5 probes, 2 ancestors, d=2.
    fields = np.zeros((5,2,2), dtype=float)
    fields[:,0,0] = [1,2,3,4,5]
    fields[:,1,1] = [5,4,3,2,1]
    bank = tmp_path/"bank.npz"
    np.savez_compressed(bank, task_fields=fields, ancestor_names=np.array(["A","B"]))
    selected = tmp_path/"sel.json"
    selected.write_text(json.dumps({"selected":[1,4]}), encoding="utf-8")
    # Only selected probe rows are present in online target/base files.
    base = tmp_path/"base.npz"; target=tmp_path/"target.npz"
    b0=np.zeros((2,1,2),dtype=float); e0=np.zeros((2,1,2),dtype=float)
    bt=np.zeros((2,1,2),dtype=float); et=np.array([[[2.,0.]], [[5.,1.]]])
    np.savez_compressed(base, base_embeddings=b0, edited_embeddings=e0)
    np.savez_compressed(target, base_embeddings=bt, edited_embeddings=et)
    out=tmp_path/"prepared.npz"
    subprocess.run([sys.executable, str(ROOT/"scripts/prepare_target_decomposition.py"), "--bank", str(bank), "--base", str(base), "--target", str(target), "--selected", str(selected), "--target-selected-only", "--output", str(out)], cwd=ROOT, check=True, capture_output=True, text=True)
    z=np.load(out)
    assert z["A"].shape == (4,2)
    assert z["y"].shape == (4,)
    assert np.allclose(z["y"], [2,0,5,1])



def test_prepare_target_selected_only_rejects_wrong_global_indices(tmp_path):
    import subprocess, json
    fields=np.zeros((5,2,1),dtype=float)
    fields[:,0,0]=[1,2,3,4,5]; fields[:,1,0]=[5,4,3,2,1]
    bank=tmp_path/'bank.npz'; np.savez_compressed(bank,task_fields=fields,ancestor_names=np.array(['A','B']))
    selected=tmp_path/'sel.json'; selected.write_text(json.dumps({'selected':[1,4]}),encoding='utf-8')
    shape=(2,1,1)
    base=tmp_path/'base.npz'; target=tmp_path/'target.npz'
    np.savez_compressed(base,base_embeddings=np.zeros(shape),edited_embeddings=np.zeros(shape),global_probe_indices=np.array([1,4]),probe_ids=np.array(['p1','p4']))
    np.savez_compressed(target,base_embeddings=np.zeros(shape),edited_embeddings=np.ones(shape),global_probe_indices=np.array([4,1]),probe_ids=np.array(['p4','p1']))
    out=tmp_path/'out.npz'
    cp=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_target_decomposition.py'),'--bank',str(bank),'--base',str(base),'--target',str(target),'--selected',str(selected),'--target-selected-only','--output',str(out)],cwd=ROOT,capture_output=True,text=True)
    assert cp.returncode != 0
    assert 'global probe indices do not match' in (cp.stderr+cp.stdout)


def test_teacher_assignment_plan_is_deterministic_and_role_correct():
    import importlib.util
    spec=importlib.util.spec_from_file_location('teacher_data',ROOT/'scripts/generate_teacher_data.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows=[{'id':str(i),'prompt':f'q{i}','domain':'math' if i%2==0 else 'code'} for i in range(20)]
    mix={'mode':'mixture','seed':7,'teachers':[{'name':'A','weight':0.7},{'name':'B','weight':0.3}]}
    p1=mod.assignment_plan(rows,mix); p2=mod.assignment_plan(rows,mix)
    assert p1==p2 and len(p1)==len(rows)
    router={'mode':'router','teachers':[{'name':'Math'},{'name':'Code'}],'domain_to_teacher':{'math':'Math','code':'Code'}}
    rp=mod.assignment_plan(rows,router)
    assert all(name==('Math' if rows[i]['domain']=='math' else 'Code') for i,name in rp)


def test_online_probe_npz_contract_mentions_global_indices():
    text=(ROOT/'scripts/run_text_probe_bank.py').read_text(encoding='utf-8')
    assert 'global_probe_indices=np.asarray(global_indices' in text
    assert "seed_index=int(row.get('global_probe_index', i))" in text



def test_probe_runner_rejects_duplicate_deterministic_replicates(tmp_path):
    import subprocess
    probes=tmp_path/'p.jsonl'; probes.write_text('{"base_query":"a","edited_query":"b"}\n',encoding='utf-8')
    cp=subprocess.run([sys.executable,str(ROOT/'scripts/run_text_probe_bank.py'),'--model','dummy','--probes',str(probes),'--encoder','dummy','--samples','2','--temperature','0','--output',str(tmp_path/'x.npz')],cwd=ROOT,capture_output=True,text=True)
    assert cp.returncode != 0
    assert 'duplicate replicates' in (cp.stdout+cp.stderr)


def test_nonlinear_job_shares_teacher_data_across_student_sizes(tmp_path):
    import subprocess,re
    job=tmp_path/'nonlinear.sh'; out=tmp_path/'cfg'; shared=tmp_path/'shared'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_full_experiment_commands.py'),'--job',str(job),'--out-dir',str(out),'--shared-out-dir',str(shared)],cwd=ROOT,check=True,capture_output=True,text=True)
    text=job.read_text(encoding='utf-8')
    assert 'RUN_L5=${RUN_L5:-1}' in text and 'RUN_L7=${RUN_L7:-1}' in text
    # At least one shared L5 teacher path must be referenced by both student-size branches.
    keys=re.findall(r'runs/teacher_data_shared/(l5teacher_[0-9a-f]+)/responses.jsonl',text)
    assert keys and any(keys.count(k)>=2 for k in set(keys))
    # L7 shared teacher stage is likewise reused across the two final student sizes.
    dkeys=re.findall(r'runs/deep_chain/(deepteacher_[0-9a-f]+)/int4_responses.jsonl',text)
    assert dkeys and any(dkeys.count(k)>=2 for k in set(dkeys))


def test_generated_audit_is_selected_only_and_transfer_labeled(tmp_path):
    import subprocess
    job=tmp_path/'audit.sh'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_audit_commands.py'),'--output',str(job)],cwd=ROOT,check=True,capture_output=True,text=True)
    text=job.read_text(encoding='utf-8')
    assert '--target-selected-only' in text
    assert '--calibration-regime g0_transfer' in text
    assert '--calibration-regime g1_transfer' in text
    assert 'runs/banks/s11/response_bank.estimate.npz' in text
    assert 'runs/calibration/s11/scores/signal_scores.csv' in text
    assert 'base_1p7b.npz' in text and 'base_4b.npz' in text
    assert '--temperature "$AUDIT_TEMPERATURE"' in text


def test_text_utility_scoring_helpers():
    import importlib.util
    spec=importlib.util.spec_from_file_location('evalutil',ROOT/'scripts/evaluate_text_utility.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.score_prediction('The answer is 42.','42','numeric') == 1.0
    assert mod.score_prediction('Final: B','B','choice') == 1.0
    assert mod.score_prediction('yes','yes','choice') == 1.0
    assert mod.score_prediction('Hello world','hello world','exact') == 1.0


def test_calibration_plan_uses_canonical_parent_labels(tmp_path):
    import subprocess,csv
    out=tmp_path/'plan'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_calibration_plan.py'),'--output-dir',str(out),'--g1-shift-n','1'],cwd=ROOT,check=True,capture_output=True,text=True)
    rows=list(csv.DictReader(open(out/'cal_support_plan.csv',encoding='utf-8',newline='')))
    assert rows
    import json
    allowed={'Math','Code','Medical','Science'}
    assert set(json.loads(rows[0]['parents'])) <= allowed


def test_generate_calibration_job_is_selected_only(tmp_path):
    import subprocess
    plan=tmp_path/'plan'; job=tmp_path/'g0.sh'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_calibration_plan.py'),'--output-dir',str(plan),'--g1-shift-n','1'],cwd=ROOT,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_calibration_commands.py'),'--plan-dir',str(plan),'--output',str(job)],cwd=ROOT,check=True,capture_output=True,text=True)
    text=job.read_text(encoding='utf-8')
    assert 'subset_probe_pool.py' in text
    assert '--target-selected-only' in text
    assert 'calibrate_support_from_prepared.py' in text
    assert '--load-in-8bit' in text and '--load-in-4bit' in text


def test_collect_subset_utilities_and_exact_shapley(tmp_path):
    import subprocess,csv,json
    plan=tmp_path/'plan.csv'; ev=tmp_path/'eval'; ev.mkdir()
    parents=['Math','Medical','Science']
    rows=[]
    import itertools
    for k in range(1,4):
        for sub in itertools.combinations(parents,k):
            cid='subset_'+'_'.join(sub);rows.append({'calibration_id':cid,'subset':';'.join(sub),'parents':json.dumps(list(sub)),'weights':'{}'})
    with plan.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['calibration_id','subset','parents','weights']);w.writeheader();w.writerows(rows)
    # Additive cooperative game: utilities are sum of parent values.
    vals={'Math':0.2,'Medical':0.5,'Science':0.3}
    def dump(name,sub):
        u=sum(vals.get(x,0) for x in sub);obj={'overall_utility':u,'n':1,'by_domain':{'ALLX':{'utility':u,'n':1}}};(ev/f'{name}.json').write_text(json.dumps(obj),encoding='utf-8')
    dump('empty',[])
    for r in rows:dump(r['calibration_id'],r['subset'].split(';'))
    raw=tmp_path/'subset.csv'; shp=tmp_path/'shap.csv'
    subprocess.run([sys.executable,str(ROOT/'scripts/collect_subset_utilities.py'),'--plan',str(plan),'--eval-dir',str(ev),'--output',str(raw)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/compute_shapley.py'),str(raw),'--output',str(shp)],cwd=ROOT,check=True)
    rr=list(csv.DictReader(open(shp,encoding='utf-8',newline='')))
    got={r['parent']:float(r['shapley']) for r in rr if r['domain']=='ALLX'}
    assert np.isclose(got['Math'],0.2) and np.isclose(got['Medical'],0.5) and np.isclose(got['Science'],0.3)


def test_modeldna_text_parser():
    import importlib.util
    spec=importlib.util.spec_from_file_location('mdna',ROOT/'scripts/run_modeldna_baseline.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    txt='''candidate                    weight   ± roles   F2 check\norg/a                        +0.650     0.003      0.646\norg/b                        +0.350     0.003      0.353\n'''
    w=mod.parse_weights(txt,['org/a','org/b'])
    assert np.isclose(w['org/a'],0.65) and np.isclose(w['org/b'],0.35)


def test_select_domain_probes_returns_global_indices(tmp_path):
    import subprocess,json
    probes=tmp_path/'probes.jsonl'
    rows=[
        {'id':'m0','domain':'Math','base_query':'a','edited_query':'b'},
        {'id':'c0','domain':'Code','base_query':'a','edited_query':'b'},
        {'id':'m1','domain':'Math','base_query':'a','edited_query':'b'},
        {'id':'m2','domain':'Math','base_query':'a','edited_query':'b'},
    ]
    probes.write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
    # [N=4,K=2,d=1], make Math rows all valid with distinct geometry.
    fields=np.array([[[1.],[0.]],[[1.],[1.]],[[0.],[1.]],[[1.],[1.]]],dtype=float)
    bank=tmp_path/'bank.npz'; np.savez_compressed(bank,task_fields=fields)
    out=tmp_path/'sel.json'
    subprocess.run([sys.executable,str(ROOT/'scripts/select_domain_probes.py'),'--bank',str(bank),'--probes',str(probes),'--domain','math','--budget','2','--output',str(out)],cwd=ROOT,check=True,capture_output=True,text=True)
    obj=json.loads(out.read_text(encoding='utf-8'))
    assert obj['candidate_count']==3
    assert len(obj['selected'])==2
    assert set(obj['selected']) <= {0,2,3}


def test_decompose_vectors_accepts_single_row_vectors_npz(tmp_path):
    import subprocess,json
    anc=tmp_path/'anc.npz'; targ=tmp_path/'targ.npz'; out=tmp_path/'out.json'
    np.savez_compressed(anc,vectors=np.eye(2),names=np.array(['A','B']))
    np.savez_compressed(targ,vectors=np.array([[0.8,0.2]]),names=np.array(['Target']))
    subprocess.run([sys.executable,str(ROOT/'scripts/decompose_vectors.py'),'--ancestors',str(anc),'--target',str(targ),'--output',str(out)],cwd=ROOT,check=True,capture_output=True,text=True)
    obj=json.loads(out.read_text(encoding='utf-8'))
    assert obj['names']==['A','B']
    assert np.allclose(obj['coordinates'],[0.8,0.2],atol=1e-8)


def test_sync_paper_results_merges_baseline_summaries(tmp_path):
    import subprocess,csv,shutil
    templates=tmp_path/'templates'; derived=tmp_path/'derived'; templates.mkdir(); derived.mkdir()
    shutil.copy(ROOT/'results/templates/main_results.csv',templates/'main_results.csv')
    shutil.copy(ROOT/'results/templates/selective_coverage.csv',templates/'selective_coverage.csv')
    shutil.copy(ROOT/'results/templates/functional_validity.csv',templates/'functional_validity.csv')
    shutil.copy(ROOT/'results/templates/open_set.csv',templates/'open_set.csv')
    fields=['method','setting','n','n_decomposable','decomposable_rate','low_signal_rate','bank_insufficient_rate','end_to_end_parent_f1_mean']
    for name,method,val in [('decomposition_summary.csv','FAS','0.90'),('dna_decomposition_summary.csv','DNA-Decomp','0.70'),('modeldna_decomposition_summary.csv','modelDNA','0.95')]:
        with (derived/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow({'method':method,'setting':'clean','n':'1','n_decomposable':'1','decomposable_rate':'1','low_signal_rate':'0','bank_insufficient_rate':'0','end_to_end_parent_f1_mean':val})
    subprocess.run([sys.executable,str(ROOT/'scripts/sync_paper_results.py'),'--templates',str(templates),'--derived',str(derived)],cwd=tmp_path,check=True,capture_output=True,text=True)
    rr=list(csv.DictReader(open(templates/'main_results.csv',encoding='utf-8',newline='')))
    got={r['method']:r['clean_f1'] for r in rr}
    assert got['FAS']=='0.9000' and got['DNA-Decomp']=='0.7000' and got['modelDNA']=='0.9500'


def test_baseline_manifest_generator_maps_scenarios_and_applicability(tmp_path):
    import subprocess,csv
    manifest=tmp_path/'runs.csv'; out=tmp_path/'baseline.csv'
    fields=['run_id','scenario','student','checkpoint_or_api','status']
    rows=[
        {'run_id':'r1','scenario':'L1-linear','student':'same','checkpoint_or_api':'/m/clean','status':'done'},
        {'run_id':'r3','scenario':'L3-merge-sft','student':'same','checkpoint_or_api':'/m/sft','status':'done'},
        {'run_id':'r5','scenario':'L5-mixture-kd','student':'1.7B','checkpoint_or_api':'/m/kd','status':'done'},
        {'run_id':'r7','scenario':'L7-deep-chain','student':'0.6B','checkpoint_or_api':'/m/deep','status':'done'},
    ]
    with manifest.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_baseline_target_manifest.py'),'--manifest',str(manifest),'--output',str(out)],cwd=ROOT,check=True,capture_output=True,text=True)
    rr={r['target_id']:r for r in csv.DictReader(out.open(encoding='utf-8',newline=''))}
    assert rr['r1']['setting']=='clean' and rr['r1']['modeldna_applicable']=='1'
    assert rr['r3']['setting']=='sft' and rr['r3']['modeldna_applicable']=='1'
    assert rr['r5']['setting']=='kd' and rr['r5']['modeldna_applicable']=='0'
    assert rr['r7']['setting']=='deep' and rr['r7']['modeldna_applicable']=='0'


def test_main_result_access_is_method_native_not_fake_common_query_unit(tmp_path):
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/build_paper_assets.py')],cwd=ROOT,check=True,capture_output=True,text=True)
    text=(ROOT/'tables/table_main_results.tex').read_text(encoding='utf-8')
    assert 'Audit access' in text
    assert '2Bm API' in text
    assert '100 prompts' in text
    assert 'weights' in text
    assert '$B\\times m$' not in text


def test_shapley_plan_preserves_full_coefficients_by_default(tmp_path):
    import subprocess,csv,json
    plan=tmp_path/'plan.csv'; job=tmp_path/'job.sh'
    subprocess.run([
        sys.executable,str(ROOT/'scripts/generate_shapley_commands.py'),
        '--parents','Math,Medical,Science','--weights','0.2,0.5,0.3',
        '--plan',str(plan),'--job',str(job)
    ],cwd=ROOT,check=True,capture_output=True,text=True)
    rows=list(csv.DictReader(plan.open(encoding='utf-8',newline='')))
    by_subset={r['subset']:json.loads(r['weights']) for r in rows}
    assert by_subset['Math']=={'Math':0.2}
    assert by_subset['Medical']=={'Medical':0.5}
    assert by_subset['Math;Medical']=={'Math':0.2,'Medical':0.5}
    assert by_subset['Math;Medical;Science']=={'Math':0.2,'Medical':0.5,'Science':0.3}
    assert all(r['weight_rule']=='fixed' for r in rows)


def test_shapley_plan_renormalized_is_explicit_sensitivity(tmp_path):
    import subprocess,csv,json
    plan=tmp_path/'plan.csv'; job=tmp_path/'job.sh'
    subprocess.run([
        sys.executable,str(ROOT/'scripts/generate_shapley_commands.py'),
        '--parents','Math,Medical,Science','--weights','0.2,0.5,0.3',
        '--subset-weight-rule','renormalized','--plan',str(plan),'--job',str(job)
    ],cwd=ROOT,check=True,capture_output=True,text=True)
    rows=list(csv.DictReader(plan.open(encoding='utf-8',newline='')))
    by_subset={r['subset']:json.loads(r['weights']) for r in rows}
    assert np.isclose(by_subset['Math;Medical']['Math'],0.2/0.7)
    assert np.isclose(by_subset['Math;Medical']['Medical'],0.5/0.7)
    assert all(r['weight_rule']=='renormalized' for r in rows)


def test_calibration_resolution_uses_realized_counts_when_provided(tmp_path):
    import subprocess,csv
    sig=tmp_path/'sig.csv'; opn=tmp_path/'open.csv'
    for path,n in [(sig,40),(opn,39)]:
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['score']);w.writeheader();w.writerows({'score':str(i/100)} for i in range(n))
    cp=subprocess.run([sys.executable,str(ROOT/'scripts/check_calibration_resolution.py'),'--actual-signal',str(sig),'--actual-open',str(opn),'--require-planned-count'],cwd=ROOT,capture_output=True,text=True)
    assert cp.returncode != 0
    assert 'realized_count_below_planned' in (cp.stdout+cp.stderr)


def test_generated_g0_job_checks_actual_calibration_counts(tmp_path):
    import subprocess
    plan=tmp_path/'plan'; job=tmp_path/'g0.sh'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_calibration_plan.py'),'--output-dir',str(plan),'--g1-shift-n','1'],cwd=ROOT,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_calibration_commands.py'),'--plan-dir',str(plan),'--output',str(job)],cwd=ROOT,check=True,capture_output=True,text=True)
    text=job.read_text(encoding='utf-8')
    assert '--actual-signal runs/calibration/scores/signal_scores.csv' in text
    assert '--actual-open runs/calibration/scores/open_scores.csv' in text
    assert '--require-planned-count' in text


def test_release_keeps_generated_paper_figures(tmp_path):
    import subprocess,zipfile
    fig=ROOT/'figures/results/__release_test_asset.txt'
    fig.parent.mkdir(parents=True,exist_ok=True);fig.write_text('paper asset',encoding='utf-8')
    out=tmp_path/'release.zip'
    try:
        subprocess.run([sys.executable,str(ROOT/'scripts/make_release.py'),'--output',str(out)],cwd=ROOT,check=True,capture_output=True,text=True)
        with zipfile.ZipFile(out) as z:
            assert 'figures/results/__release_test_asset.txt' in z.namelist()
    finally:
        fig.unlink(missing_ok=True)


def test_v09_evidence_tiers_core_is_minimal_and_preregistered(tmp_path):
    import subprocess,csv
    full=tmp_path/'full.csv'; tiered=tmp_path/'tiered.csv'; tiers=tmp_path/'tiers'
    subprocess.run([sys.executable,str(ROOT/'scripts/expand_experiment_matrix.py'),'--output',str(full)],cwd=ROOT,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/assign_evidence_tiers.py'),'--manifest',str(full),'--output',str(tiered),'--out-dir',str(tiers)],cwd=ROOT,check=True,capture_output=True,text=True)
    rows=list(csv.DictReader(tiered.open(encoding='utf-8',newline='')))
    core=[r for r in rows if r['evidence_tier']=='core']
    assert len(rows)==54 and len(core)==15
    assert sum(r['scenario']=='L8-open-set' for r in core)==3
    assert {r['seed'] for r in core if r['scenario']=='L8-open-set'}=={'11'}
    assert all(r.get('student')!='0p6b' for r in core)


def test_formal_expert_configs_are_seed_specific(tmp_path):
    import subprocess,yaml
    out=tmp_path/'cfg'; job=tmp_path/'experts.sh'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_seeded_expert_configs.py'),'--out-dir',str(out),'--job',str(job)],cwd=ROOT,check=True,capture_output=True,text=True)
    for seed in [11,23,47]:
        for dom in ['math','code','medical','science']:
            cfg=yaml.safe_load((out/f's{seed}/{dom}.yaml').read_text(encoding='utf-8'))
            assert cfg['seed']==seed
            assert cfg['output_dir']==f'runs/experts/s{seed}/{dom}'
    text=job.read_text(encoding='utf-8')
    assert 'runs/experts/s11/math/adapter' in text
    assert 'runs/experts/math/adapter' not in text


def test_formal_bank_job_builds_independent_seed_banks(tmp_path):
    import subprocess
    job=tmp_path/'banks.sh'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_seeded_bank_commands.py'),'--output',str(job)],cwd=ROOT,check=True,capture_output=True,text=True)
    text=job.read_text(encoding='utf-8')
    for seed in [11,23,47]:
        assert f'runs/banks/s{seed}/response_bank.select.npz' in text
        assert f'runs/banks/s{seed}/response_bank.estimate.npz' in text
        assert f'runs/banks/s{seed}/selected_probes.json' in text
        assert f'runs/experts/s{seed}/math/adapter' in text


def test_nonlinear_core_uses_seed_specific_formal_ancestors(tmp_path):
    import subprocess,csv
    full=tmp_path/'full.csv'; tiered=tmp_path/'tiered.csv'; tiers=tmp_path/'tiers'; job=tmp_path/'nonlinear.sh'; out=tmp_path/'out'; shared=tmp_path/'shared'
    subprocess.run([sys.executable,str(ROOT/'scripts/expand_experiment_matrix.py'),'--output',str(full)],cwd=ROOT,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/assign_evidence_tiers.py'),'--manifest',str(full),'--output',str(tiered),'--out-dir',str(tiers)],cwd=ROOT,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_full_experiment_commands.py'),'--manifest',str(tiers/'core.csv'),'--job',str(job),'--out-dir',str(out),'--shared-out-dir',str(shared)],cwd=ROOT,check=True,capture_output=True,text=True)
    text=job.read_text(encoding='utf-8')
    for seed in [11,23,47]: assert f'runs/experts/s{seed}/math/adapter' in text
    assert 'runs/experts/math/adapter' not in text
    assert 'deepteacher_' in text


def test_open_set_core_restricts_verifier_bank_and_withholds_science(tmp_path):
    import subprocess
    job=tmp_path/'open.sh'
    subprocess.run([sys.executable,str(ROOT/'scripts/generate_open_set_commands.py'),'--seed','11','--output',str(job)],cwd=ROOT,check=True,capture_output=True,text=True)
    text=job.read_text(encoding='utf-8')
    assert '--names Math,Code,Medical' in text
    assert 'unknown_A' in text and 'Science' in text
    assert 'runs/experts/s11/science/adapter' in text
    assert 'L7-deep-ancestry__v1__1p7b__s11' in text
    assert '--calibration-regime matched' in text
    assert '--calibration-regime g1_transfer' in text


def test_diffusion_probe_runner_emits_common_fas_schema_contract():
    text=(ROOT/'scripts/run_diffusion_probe_bank.py').read_text(encoding='utf-8')
    for token in ['base_embeddings=', 'edited_embeddings=', 'probe_ids=', 'global_probe_indices=']:
        assert token in text
    assert 'paired latent seed' in text


def test_data_protocol_sources_are_disjoint_and_balanced():
    import subprocess
    cp=subprocess.run([sys.executable,str(ROOT/'scripts/check_data_protocol.py')],cwd=ROOT,check=True,capture_output=True,text=True)
    assert 'DATA_PROTOCOL_OK' in cp.stdout


def test_data_governance_overlap_and_mcq_helpers():
    from fas_core.data_governance import normalize_text, word_ngrams, contamination_candidates, max_overlap, format_mcq
    assert normalize_text('  Hello, WORLD!! ')=='hello world'
    p,a=format_mcq('Q?',['x','y'],1)
    assert 'A. x' in p and a.startswith('B.')
    refs=[{'prompt':'A train-like prompt about photosynthesis and chlorophyll.'}]
    ref,inv=contamination_candidates(refs,3)
    z=max_overlap('A train-like prompt about photosynthesis and chlorophyll.',ref,inv,3)
    assert z['exact'] and z['jaccard']==1.0


def test_formal_transform_stem_filters_images_and_medmcqa_is_stable():
    import importlib.util
    spec=importlib.util.spec_from_file_location('prep_formal',ROOT/'scripts/prepare_formal_datasets.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    bad={'subject':'science','pic_prob':True,'pic_choice':False,'problem':'x','choices':['a','b'],'answer_idx':0}
    assert mod.transform(bad,'stem_science_text','Science') is None
    good={'subject':'science','pic_prob':False,'pic_choice':False,'problem':'sky?','choices':['blue','red'],'answer_idx':0,'grade':'3','skill':'color'}
    r=mod.transform(good,'stem_science_text','Science'); assert r and r['response'].startswith('A.')
    m={'question':'Q','opa':'a','opb':'b','opc':'c','opd':'d','cop':2,'exp':'because'}
    r=mod.transform(m,'medmcqa','Medical'); assert r['response'].startswith('C. c')


def test_contamination_audit_reports_pretrim_counts(tmp_path):
    import subprocess,json
    train=tmp_path/'raw';train.mkdir();refs=tmp_path/'refs.jsonl';out=tmp_path/'out';report=tmp_path/'report.json'
    refs.write_text(json.dumps({'prompt':'exact overlap prompt alpha beta gamma delta epsilon zeta eta theta'})+'\n',encoding='utf-8')
    rows=[{'id':'bad','prompt':'exact overlap prompt alpha beta gamma delta epsilon zeta eta theta','response':'r'}]
    rows += [{'id':f'g{i}','prompt':f'unique training prompt {i} lorem ipsum dolor sit amet consectetur','response':'ok'} for i in range(5)]
    (train/'math.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_data_contamination.py'),'--train-dir',str(train),'--references',str(refs),'--output-dir',str(out),'--report',str(report),'--final-per-domain','3'],cwd=ROOT,check=True,capture_output=True,text=True)
    z=json.loads(report.read_text())['domains']['math']
    assert z['input']==6 and z['removed_contamination']==1 and z['clean_before_trim']==5 and z['kept']==3 and z['trimmed_clean']==2


def test_formal_intervention_review_gate_is_sample_based(tmp_path):
    import subprocess,json,csv
    cand=tmp_path/'cand.jsonl';review=tmp_path/'review.tsv';out=tmp_path/'final.jsonl'
    rows=[]
    for d in ['Math','Code','Medical','Science']:
        for i in range(20): rows.append({'probe_id':f'{d}_{i}','domain':d,'type':'x','base_query':'q','edited_query':'q2','needs_review':True})
    cand.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    with review.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['probe_id','domain','type','needs_review','base_query','edited_query','decision','reviewer_note'],delimiter='\t');w.writeheader()
        for r in rows[:]: w.writerow({**r,'decision':'accept','reviewer_note':''})
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_formal_interventions.py'),'--input',str(cand),'--review',str(review),'--output',str(out),'--min-per-domain','10'],cwd=ROOT,check=True,capture_output=True,text=True)
    assert len(out.read_text().splitlines())==80

def test_partitioned_logdet_respects_groups():
    import numpy as np
    from fas_core.active_probe import greedy_logdet_select_partitioned
    blocks=[np.eye(2)*s for s in [5,4,3,2,1,1]]
    groups=['a','a','a','b','b','b']
    sel,_=greedy_logdet_select_partitioned(blocks,4,groups)
    assert sum(groups[i]=='a' for i in sel)==2
    assert sum(groups[i]=='b' for i in sel)==2

def test_balanced_selector_even_default_quota():
    import numpy as np
    from fas_core.active_probe import greedy_logdet_select_partitioned
    blocks=[np.array([[10.,0.],[0.,1.]]),np.array([[9.,0.],[0.,1.]]),np.array([[1.,0.],[0.,4.]]),np.array([[1.,0.],[0.,3.]])]
    groups=['A','A','B','B']
    sel,_=greedy_logdet_select_partitioned(blocks,2,groups)
    assert {groups[i] for i in sel}=={'A','B'}

def test_proxy_domain_generator_uses_local_rng_not_global_random():
    text=(ROOT/'scripts/proxy/run_tiny_neural_proxy.py').read_text(encoding='utf-8')
    assert 'random.choice([' not in text
    assert 'rng.choice([' in text

def test_transformer_capacity_proxy_aggregation_respects_requested_specs():
    text=(ROOT/'scripts/proxy/run_transformer_capacity_proxy.py').read_text(encoding='utf-8')
    assert text.count('if label not in want: continue') >= 2
