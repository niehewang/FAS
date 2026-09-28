#!/usr/bin/env python3
"""Generate resumable L5/L6/L7 configs and staged shell execution plans.

Formal runs are genealogy-seed aware:
* each run uses ancestors under runs/experts/s<seed>/<domain>/adapter;
* L5/L6 teacher data are shared only when seed, assignment and prompt corpus match;
* L7 materialized parents and merge keys are seed-specific;
* student-size branches reuse the same seed-matched teacher data/stages.

Pilot convenience experts under runs/experts/<domain>/adapter are intentionally
not used here.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
PARENT_NAMES=['math','code','medical','science']
PARENT_LABEL={'math':'Math','code':'Code','medical':'Medical','science':'Science'}

def dump_yaml(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(yaml.safe_dump(obj,sort_keys=False,allow_unicode=True),encoding='utf-8')

def parse_json(s):
    s=str(s or '').strip(); return json.loads(s) if s else []

def shell_q(s): return "'"+str(s).replace("'","'\\''")+"'"

def cmd_path(p):
    p=Path(p)
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)

def expert_adapter(seed,parent): return f'runs/experts/s{seed}/{parent}/adapter'
def expert_full(seed,parent): return f'runs/full/s{seed}/{parent}'

def ensure_teacher_specs(parents,seed,weights=None):
    weights=weights or [None]*len(parents); out=[]
    for p,w in zip(parents,weights):
        x={'name':PARENT_LABEL[p],'base':'Qwen/Qwen3-4B-Base','adapter':expert_adapter(seed,p)}
        if w is not None:x['weight']=float(w)
        out.append(x)
    return out

def student_model(student): return {'1p7b':'Qwen/Qwen3-1.7B-Base','0p6b':'Qwen/Qwen3-0.6B-Base'}[student]

def student_cfg(student,data_path,seed,outdir,domain='MultiTeacherStudent'):
    return {'model_name':student_model(student),'domain':domain,'output_dir':outdir,'seed':int(seed),'trust_remote_code':True,
            'torch_dtype':'bfloat16','qlora_4bit':True,'bnb_4bit_quant_type':'nf4','bnb_compute_dtype':'bfloat16','bnb_double_quant':True,
            'gradient_checkpointing':True,'max_seq_length':2048,
            'data':{'path':data_path,'split':'train','prompt_field':'prompt','response_field':'response','mask_prompt_loss':True,'use_chat_template':False,'template':'{prompt}\n{response}'},
            'lora':{'r':16,'alpha':32,'dropout':0.05,'bias':'none','target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']},
            'training':{'epochs':1.0,'max_steps':-1,'batch_size':1,'grad_accum':16,'learning_rate':2e-4,'weight_decay':0.0,'warmup_ratio':0.03,'lr_scheduler':'cosine','logging_steps':10,'save_strategy':'steps','save_steps':250,'save_total_limit':2,'bf16':True,'fp16':False,'report_to':'none'}}

def sft_cfg(model_name,seed,outdir,data_path):
    c=student_cfg('1p7b',data_path,seed,outdir,domain='AuxiliarySFT'); c['model_name']=model_name;c['training']['learning_rate']=1e-4;return c

def stable_key(payload,prefix):
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:12]
    return f'{prefix}_{h}'

def add_metadata(lines,run_id,config_path,checkpoint,parents):
    lines += [f"python scripts/record_run_metadata.py --run-id {shell_q(run_id)} --config {shell_q(config_path)} --checkpoint {shell_q(checkpoint)} --parents {shell_q(';'.join(parents))} --output {shell_q(f'runs/metadata/{run_id}.json')}",
              f"python scripts/manifest_status.py set {shell_q(run_id)} ready --checkpoint-or-api {shell_q(checkpoint)}"]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',default='data/metadata/experiment_run_manifest.csv');ap.add_argument('--out-dir',default='configs/generated_runs');ap.add_argument('--shared-out-dir',default='configs/generated_shared');ap.add_argument('--job',default='jobs/run_nonlinear_experiments.sh');ap.add_argument('--mixture-prompts',default='data/distillation/pilot_prompts.jsonl');ap.add_argument('--router-prompts',default='data/distillation/pilot_prompts_with_domains.jsonl');ap.add_argument('--aux-sft',default='data/deep_chain/auxiliary_sft.jsonl');ap.add_argument('--deep-kd-prompts',default='data/distillation/pilot_prompts.jsonl');args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.manifest).open(encoding='utf-8',newline='')));rows=[r for r in rows if r['scenario'].startswith(('L5-','L6-','L7-'))]
    out_root=ROOT/args.out_dir;shared_root=ROOT/args.shared_out_dir;out_root.mkdir(parents=True,exist_ok=True);shared_root.mkdir(parents=True,exist_ok=True)
    job=ROOT/args.job;job.parent.mkdir(parents=True,exist_ok=True)
    L=['#!/usr/bin/env bash','set -euo pipefail','cd "$(dirname "$0")/.."','',
       '# Auto-generated formal nonlinear ancestry pipeline (seed-specific ancestor families).',
       'RUN_L5=${RUN_L5:-1}','RUN_L6=${RUN_L6:-1}','RUN_L7=${RUN_L7:-1}',
       'mkdir -p runs/{metadata,checkpoints,teacher_data_shared,deep_chain,full}','']

    generated=0
    for r in rows:
        run_id=r['run_id'];scenario=r['scenario'];seed=int(r['seed']);student=r['student'];parents=[x for x in r['parents'].split(';') if x];variant=r['variant']
        rd=out_root/run_id;rd.mkdir(parents=True,exist_ok=True)
        gate='RUN_L5' if scenario.startswith('L5-') else ('RUN_L6' if scenario.startswith('L6-') else 'RUN_L7')
        block=[f'if [ "${{{gate}}}" = "1" ]; then',f'  echo "=== {run_id} ==="',f'  mkdir -p {shell_q(f"runs/checkpoints/{run_id}")}']
        for p in parents:
            block += [f'  test -d {expert_adapter(seed,p)} || {{ echo "missing formal ancestor {expert_adapter(seed,p)}"; exit 2; }}']
        if scenario.startswith('L5-'):
            exposure=parse_json(r['exposure'])
            key=stable_key({'scenario':'L5','variant':variant,'seed':seed,'parents':parents,'exposure':exposure,'prompts':args.mixture_prompts},'l5teacher')
            shared_data=f'runs/teacher_data_shared/{key}/responses.jsonl'; shared_cfg=shared_root/f'{key}.yaml'
            tcfg={'mode':'mixture','seed':seed,'input_jsonl':args.mixture_prompts,'output_jsonl':shared_data,'teachers':ensure_teacher_specs(parents,seed,exposure),'generation':{'max_new_tokens':512,'do_sample':False}}
            dump_yaml(shared_cfg,tcfg)
            scfg=student_cfg(student,shared_data,seed,f'runs/checkpoints/{run_id}/student'); sp=rd/'student.yaml';dump_yaml(sp,scfg)
            block += [f'  mkdir -p {shell_q(str(Path(shared_data).parent))}',f'  if [ ! -s {shell_q(shared_data)} ]; then python scripts/generate_teacher_data.py --config {shell_q(cmd_path(shared_cfg))}; fi',f'  if [ ! -d {shell_q(scfg["output_dir"]+"/adapter")} ]; then python scripts/train_lora_expert.py --config {shell_q(cmd_path(sp))}; fi']
            add_metadata(block,run_id,cmd_path(sp),scfg['output_dir']+'/adapter',parents)
        elif scenario.startswith('L6-'):
            key=stable_key({'scenario':'L6','variant':variant,'seed':seed,'parents':parents,'router':r.get('router',''),'prompts':args.router_prompts},'l6teacher')
            shared_data=f'runs/teacher_data_shared/{key}/responses.jsonl'; shared_cfg=shared_root/f'{key}.yaml'
            tcfg={'mode':'router','seed':seed,'input_jsonl':args.router_prompts,'output_jsonl':shared_data,'teachers':ensure_teacher_specs(parents,seed),'domain_to_teacher':{PARENT_LABEL[p]:PARENT_LABEL[p] for p in parents},'generation':{'max_new_tokens':512,'do_sample':False}}
            dump_yaml(shared_cfg,tcfg)
            student_out=f'runs/checkpoints/{run_id}/student_router';scfg=student_cfg(student,shared_data,seed,student_out);sp=rd/'student.yaml';dump_yaml(sp,scfg)
            full=f'runs/checkpoints/{run_id}/student_router_full';aux_out=f'runs/checkpoints/{run_id}/student_router_sft';acfg=sft_cfg(full,seed,aux_out,args.aux_sft);apath=rd/'aux_sft.yaml';dump_yaml(apath,acfg)
            block += [f'  mkdir -p {shell_q(str(Path(shared_data).parent))}',f'  if [ ! -s {shell_q(shared_data)} ]; then python scripts/generate_teacher_data.py --config {shell_q(cmd_path(shared_cfg))}; fi',f'  if [ ! -d {shell_q(student_out+"/adapter")} ]; then python scripts/train_lora_expert.py --config {shell_q(cmd_path(sp))}; fi',f'  if [ ! -f {shell_q(full+"/config.json")} ]; then python scripts/materialize_lora.py --base {shell_q(student_model(student))} --adapter {shell_q(student_out+"/adapter")} --output {shell_q(full)}; fi',f'  if [ ! -d {shell_q(aux_out+"/adapter")} ]; then python scripts/train_lora_expert.py --config {shell_q(cmd_path(apath))}; fi']
            add_metadata(block,run_id,cmd_path(apath),aux_out+'/adapter',parents)
        else:  # L7
            for p in parents:
                block += [f'  if [ ! -f {expert_full(seed,p)}/config.json ]; then mkdir -p runs/full/s{seed}; python scripts/materialize_lora.py --base Qwen/Qwen3-4B-Base --adapter {expert_adapter(seed,p)} --output {expert_full(seed,p)}; fi']
            weights=parse_json(r['weights']); merge_key=stable_key({'parents':parents,'weights':weights,'seed':seed},'merge')
            merged=f'runs/deep_chain/{merge_key}'
            merge_cfg={'merge_method':'dare_ties','base_model':'Qwen/Qwen3-4B-Base','models':[{'model':expert_full(seed,p),'parameters':{'weight':float(w),'density':0.5}} for p,w in zip(parents,weights)],'parameters':{'normalize':True},'tokenizer_source':'base','dtype':'bfloat16'}
            merge_cfg_path=shared_root/f'{merge_key}.yaml';dump_yaml(merge_cfg_path,merge_cfg)
            deep_key=stable_key({'merge_key':merge_key,'seed':seed,'aux':args.aux_sft,'kd_prompts':args.deep_kd_prompts},'deepteacher')
            shared_dir=f'runs/deep_chain/{deep_key}';sft_out=f'{shared_dir}/sft';sft_full=f'{shared_dir}/sft_full';shared_data=f'{shared_dir}/int4_responses.jsonl'
            sfcfg=sft_cfg(merged,seed,sft_out,args.aux_sft);sfp=shared_root/f'{deep_key}_sft.yaml';dump_yaml(sfp,sfcfg)
            tcfg={'mode':'mixture','seed':seed,'input_jsonl':args.deep_kd_prompts,'output_jsonl':shared_data,'teachers':[{'name':'DeepChainTeacherINT4','model':sft_full,'weight':1.0,'load_in_4bit':True,'bnb_4bit_quant_type':'nf4','dtype':'bfloat16'}],'generation':{'max_new_tokens':512,'do_sample':False}}
            tp=shared_root/f'{deep_key}_teacher.yaml';dump_yaml(tp,tcfg)
            student_out=f'runs/checkpoints/{run_id}/deep_student';stcfg=student_cfg(student,shared_data,seed,student_out);stp=rd/'student.yaml';dump_yaml(stp,stcfg)
            block += [f'  if [ ! -f {shell_q(merged+"/config.json")} ]; then mergekit-yaml {shell_q(cmd_path(merge_cfg_path))} {shell_q(merged)} --cuda --lazy-unpickle; fi',f'  if [ ! -d {shell_q(sft_out+"/adapter")} ]; then python scripts/train_lora_expert.py --config {shell_q(cmd_path(sfp))}; fi',f'  if [ ! -f {shell_q(sft_full+"/config.json")} ]; then python scripts/materialize_lora.py --base {shell_q(merged)} --adapter {shell_q(sft_out+"/adapter")} --output {shell_q(sft_full)}; fi',f'  if [ ! -s {shell_q(shared_data)} ]; then python scripts/generate_teacher_data.py --config {shell_q(cmd_path(tp))}; fi',f'  if [ ! -d {shell_q(student_out+"/adapter")} ]; then python scripts/train_lora_expert.py --config {shell_q(cmd_path(stp))}; fi']
            add_metadata(block,run_id,cmd_path(stp),student_out+'/adapter',parents)
        block += ['fi','']; L += block; generated+=1
    L += ['echo "Formal nonlinear ancestry construction finished. Audit with seed-matched banks."']
    job.write_text('\n'.join(L)+'\n',encoding='utf-8');job.chmod(0o755)
    print(f'generated configs for {generated} runs -> {cmd_path(out_root)}');print(cmd_path(job))
if __name__=='__main__':main()
