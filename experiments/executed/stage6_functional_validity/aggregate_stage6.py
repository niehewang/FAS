#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
SEEDS=[23,47,71];P=['Math','Medical','Science'];DOMAINS=P;W={'Math':0.2,'Medical':0.5,'Science':0.3};EPS=1e-12

def key(s):return frozenset(x for x in (s or '').split(';') if x)
def exact_shapley(U):
    out={p:0. for p in P};fact=math.factorial;n=len(P)
    for p in P:
      rest=[x for x in P if x!=p]
      for r in range(len(rest)+1):
       for c in itertools.combinations(rest,r):
        S=frozenset(c);Sp=frozenset(set(S)|{p});out[p]+=fact(len(S))*fact(n-len(S)-1)/fact(n)*(U[Sp]-U[S])
    return out

def rho(x,y):
    z=spearmanr(x,y).statistic
    return float(z) if np.isfinite(z) else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True);ap.add_argument('--output-json',required=True);ap.add_argument('--output-rows',required=True);a=ap.parse_args();R=Path(a.run_root);cells=[];parent_rows=[];seed_summaries=[]
    for s in SEEDS:
        util=list(csv.DictReader(open(R/f's{s}/utility/subset_utilities.csv',encoding='utf-8',newline='')));coord=json.loads((R/f's{s}/coordinates.json').read_text()); by={(r['domain'],key(r['subset'])):float(r['utility']) for r in util}
        ss=[]
        for d in DOMAINS:
            U={S:v for (dd,S),v in by.items() if dd==d};need={frozenset(c) for k in range(4) for c in itertools.combinations(P,k)}
            if set(U)!=need:raise ValueError(f's{s} {d}: utility subsets mismatch {len(U)} vs 8')
            sh=exact_shapley(U);full=frozenset(P);loo={p:U[full]-U[full-{p}] for p in P};eff=sum(sh.values())-(U[full]-U[frozenset()])
            dd=coord['domains'][d];cand=coord['ancestor_names'];fm={p:float(dd['fas_pi'][cand.index(p)]) for p in P};am={p:float(dd['absolute_pi'][cand.index(p)]) for p in P}
            vals={'Merge weights':W,'Absolute-output decomposition':am,'FAS coordinates':fm}
            cr={q:rho([m[p] for p in P],[sh[p] for p in P]) for q,m in vals.items()};lr={q:rho([m[p] for p in P],[loo[p] for p in P]) for q,m in vals.items()}
            top_sh=max(P,key=lambda p:sh[p]);tops={q:max(P,key=lambda p:m[p]) for q,m in vals.items()}
            cell={'seed':s,'domain':d,'shapley':sh,'leave_one_out':loo,'utility_empty':U[frozenset()],'utility_full':U[full],'shapley_efficiency_residual':eff,'quantities':vals,'spearman_shapley':cr,'spearman_loo':lr,'top_shapley_parent':top_sh,'top_parent_by_quantity':tops,'fas_false_candidate_code_mass':dd['fas_false_candidate_code_mass'],'absolute_false_candidate_code_mass':dd['absolute_false_candidate_code_mass'],'fas_relative_residual':dd['fas_relative_residual'],'absolute_relative_residual':dd['absolute_relative_residual']}
            cells.append(cell);ss.append(cell)
            for p in P: parent_rows.append({'seed':s,'domain':d,'parent':p,'shapley':sh[p],'loo':loo[p],'construction_weight':W[p],'fas_coordinate':fm[p],'absolute_coordinate':am[p]})
        seed_summaries.append({'seed':s,'mean_fas_spearman':float(np.mean([c['spearman_shapley']['FAS coordinates'] for c in ss if c['spearman_shapley']['FAS coordinates'] is not None])),'mean_construction_spearman':float(np.mean([c['spearman_shapley']['Merge weights'] for c in ss if c['spearman_shapley']['Merge weights'] is not None]))})
    qnames=['Merge weights','Absolute-output decomposition','FAS coordinates'];agg={}
    for q in qnames:
        vs=[c['spearman_shapley'][q] for c in cells if c['spearman_shapley'][q] is not None];vl=[c['spearman_loo'][q] for c in cells if c['spearman_loo'][q] is not None]
        agg[q]={'mean_spearman_shapley':float(np.mean(vs)) if vs else None,'std_spearman_shapley':float(np.std(vs,ddof=1)) if len(vs)>1 else 0.0,'n_cells':len(vs),'mean_spearman_loo':float(np.mean(vl)) if vl else None,'top1_shapley_match_rate':float(np.mean([c['top_parent_by_quantity'][q]==c['top_shapley_parent'] for c in cells]))}
    wins=sum((c['spearman_shapley']['FAS coordinates'] is not None and c['spearman_shapley']['Merge weights'] is not None and c['spearman_shapley']['FAS coordinates']>=c['spearman_shapley']['Merge weights']) for c in cells)
    mf=agg['FAS coordinates']['mean_spearman_shapley'];mc=agg['Merge weights']['mean_spearman_shapley'];gate=bool(mf is not None and mc is not None and mf-mc>=0.10 and wins>=6 and mf>=0.0)
    out={'protocol':'FAS_STAGE6_FUNCTIONAL_VALIDITY_EXACT_SHAPLEY_V1','cells':cells,'seed_summaries':seed_summaries,'aggregate':agg,'diagnostics':{'fas_win_or_tie_cells_vs_construction':wins,'mean_fas_minus_construction_spearman':None if mf is None or mc is None else mf-mc,'mean_fas_false_candidate_code_mass':float(np.mean([c['fas_false_candidate_code_mass'] for c in cells])),'mean_absolute_false_candidate_code_mass':float(np.mean([c['absolute_false_candidate_code_mass'] for c in cells])),'max_abs_shapley_efficiency_residual':float(max(abs(c['shapley_efficiency_residual']) for c in cells))},'gate':{'rule':'mean(FAS-Shapley rho)-mean(construction-Shapley rho)>=0.10; FAS wins/ties construction in >=6/9 cells; mean FAS rho>=0','pass':gate,'claim_if_pass':'C4 supported in tested controlled same-scale adapter-composition family; FAS ranking aligns with realized counterfactual contribution better than construction share on average, without equating FAS to causal contribution.','claim_if_fail':'C4 downgraded; FAS remains a geometric ancestry coordinate and functional-contribution interpretation is unsupported.'},'guardrails':['Exact Shapley uses fixed-coefficient 2^3 subset game.','Negative Shapley values are retained.','Seed 11 excluded.','Stage5 cross-scale failure is not reinterpreted by Stage6.']}
    Path(a.output_json).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    with open(a.output_rows,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','domain','parent','shapley','loo','construction_weight','fas_coordinate','absolute_coordinate']);w.writeheader();w.writerows(parent_rows)
    print(json.dumps({'aggregate':agg,'diagnostics':out['diagnostics'],'gate':out['gate']},indent=2))
if __name__=='__main__':main()
