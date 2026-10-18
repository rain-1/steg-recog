"""Submit/status/fetch HF generation jobs, with private allowlisted bundles.

Use python -m scripts.hf_pilot --config <locked-config> submit|status|fetch.
Default is the configured pilot (64 exposure + 64 behavior sources in the
answer-only design). --stage generate requires both pilot gates and
a measured cost projection. Training is never launched on HF.
"""
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import shutil
import tempfile
import time
import uuid

from wmscale.common import load_config, out_dir, file_sha256, write_json

IMAGE='vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3'
FLAVOR='a100-large'
TIMEOUT_SECONDS=3600
MAX_HOURLY_USD=2.51
TERMINAL={'COMPLETED','ERROR','CANCELED','DELETED'}


def read(path):return json.loads(Path(path).read_text())


def bundle_files(repo,root,dest,*,stage='pilot'):
    code=dest/'code'
    for package in ['wmscale','wmsteg']:
        for src in (repo/package).glob('*.py'):
            if src.is_symlink():raise ValueError('Bundle refuses symlinked source files')
            target=code/package/src.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,target)
    worker=code/'scripts/hf_pilot_worker.py';worker.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(repo/'scripts/hf_pilot_worker.py',worker)
    for name in ['manifest.json','config.lock.json','environment.json','tinker_capabilities.json','reference_precheck.json']:
        src=root/name
        if src.exists():
            target=dest/'run'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    # Only prepared public prompt pools, never .env, caches, user files or keys.
    for src in (root/'prompts').glob('*.json*'):
        target=dest/'run/prompts'/src.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    if not (dest/'run/prompts/summary.json').exists():raise ValueError('Prepare prompts first')
    if stage=='generate':
        for name in ['pilot_gate.json','pilot_review.jsonl','semantic_review.json','neutral_ids.json','cost_projection.json']:
            if (root/name).exists():shutil.copy2(root/name,dest/'run'/name)
        for src in (root/'gen').rglob('*.json*'):
            if src.is_symlink():raise ValueError('Bundle refuses symlinked generation artifacts')
            target=dest/'run/gen'/src.relative_to(root/'gen')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    return {str(p.relative_to(dest)):file_sha256(p) for p in dest.rglob('*') if p.is_file()}


def submit(cfg,api,*,stage='pilot'):
    from huggingface_hub import Volume
    root=out_dir(cfg);record=root/('hf_job.json' if stage=='pilot' else 'hf_generate_job.json')
    if record.exists():raise ValueError('A job is already recorded; inspect/fetch it before submitting another')
    require_reference_precheck(cfg,root)
    hardware=next(x for x in api.list_jobs_hardware() if x.name==FLAVOR)
    if hardware.unit_label!='minute':raise ValueError('Unexpected HF pricing unit')
    hourly=hardware.unit_cost_usd*60
    if not 0<hourly<=MAX_HOURLY_USD:raise ValueError('GPU price exceeds launch cap')
    timeout=TIMEOUT_SECONDS
    if stage=='generate':
        from wmscale.preflight import estimate
        projection=estimate(cfg,hourly,enforce_generation_budget=True)
        timeout=math.ceil(projection['planning_generation_hours_with_2x_margin']*3600)+900
        if timeout>8*3600:raise ValueError('Projected generation exceeds the eight-hour single-job limit')
    # Round billing up a minute and include $0.10 for the small artifact bucket.
    reserve=(math.ceil(timeout/60)+1)*hardware.unit_cost_usd+.10
    ledger=Path(cfg['execution']['budget_ledger']);ledger.parent.mkdir(parents=True,exist_ok=True)
    with open(ledger.with_suffix('.lock'),'w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=read(ledger) if ledger.exists() else {'entries':[]}
        used=sum(x.get('cost_usd',x['reserved_usd']) for x in data['entries'])
        training_reserve=cfg['execution'].get('training_reserve_usd',35) if stage=='generate' else 0
        if used+reserve+training_reserve>min(cfg['execution']['phase_cap_usd'],cfg['execution']['total_cap_usd']):
            raise ValueError('Insufficient shared budget for HF pilot')
        entry={'label':f'HF {stage}','run_name':cfg['run_name'],'backend':'hf_jobs',
               'reserved_usd':reserve,'hourly_usd':hourly,'status':'reserved','started_unix':time.time()}
        data['entries'].append(entry);write_json(ledger,data)
    namespace=api.whoami()['name']
    bucket=f'{namespace}/steg-recog-pilot-{uuid.uuid4().hex[:10]}'
    api.create_bucket(bucket,private=True)
    if not api.bucket_info(bucket).private:raise ValueError('Artifact bucket must be private')
    # Record the reservation before submission; an uncertain network failure must
    # be inspected, not automatically resubmitted and potentially double-billed.
    metadata={'bucket':bucket,'namespace':namespace,'image':IMAGE,'flavor':FLAVOR,
              'stage':stage,'timeout_seconds':timeout,'reserved_usd':reserve,'hourly_usd':hourly,'job_id':None}
    write_json(record,metadata)
    with tempfile.TemporaryDirectory(prefix='steg-recog-hf-') as tmp:
        dest=Path(tmp)
        metadata['bundle_hashes']=bundle_files(Path(__file__).resolve().parents[1],root,dest,stage=stage)
        shutil.copy2(ledger,dest/'budget.json')
        metadata['bundle_hashes']['budget.json']=file_sha256(dest/'budget.json')
        api.sync_bucket(str(dest),f'hf://buckets/{bucket}/bundle',quiet=True)
    api.batch_bucket_files(bucket,add=[(b'', 'outputs/.keep')])
    write_json(record,metadata)
    job=api.run_job(image=IMAGE,command=['python3','/bundle/code/scripts/hf_pilot_worker.py'],
        flavor=FLAVOR,timeout=timeout,namespace=namespace,name=cfg['run_name']+'-'+stage,
        env={'GPU_HOURLY_USD':str(hourly),'HF_STAGE':stage,'PYTHONUNBUFFERED':'1','WANDB_DISABLED':'true'},
        secrets={'HF_TOKEN':os.environ['HF_TOKEN']},
        volumes=[Volume(type='bucket',source=bucket,path='bundle',mount_path='/bundle',read_only=True),
                 Volume(type='bucket',source=bucket,path='outputs',mount_path='/outputs',read_only=False)])
    metadata.update(job_id=job.id,url=job.url)
    write_json(record,metadata)
    print(json.dumps({k:metadata[k] for k in ['job_id','url','reserved_usd','timeout_seconds']},indent=2))


def require_reference_precheck(cfg,root):
    """Known reference failures must be resolved before any paid submission."""
    if not cfg.get('prompts',{}).get('reference_review_required'):return
    from wmscale.common import read_jsonl
    review=read(root/'reference_precheck.json')
    path=root/'prompts/exposure.jsonl';rows=read_jsonl(path)[:cfg['execution']['pilot_sources']]
    decisions=review.get('decisions',[])
    if (review.get('source_hash')!=file_sha256(path) or not review.get('reviewer')
            or len(decisions)!=len(rows) or {d.get('id') for d in decisions}!={r['id'] for r in rows}
            or any(d.get('acceptable') is not True or not d.get('note') for d in decisions)):
        raise ValueError('Missing, stale or failed reference precheck; review every pilot reference before spending')


def status(cfg,api,fetch=False,*,stage='pilot'):
    root=out_dir(cfg);meta=read(root/('hf_job.json' if stage=='pilot' else 'hf_generate_job.json'))
    if not meta['job_id']:raise ValueError('Submission status uncertain: inspect HF Jobs before retrying')
    job=api.inspect_job(job_id=meta['job_id'],namespace=meta['namespace'])
    result={'job_id':job.id,'url':job.url,'stage':job.status.stage,'message':job.status.message,
            'reserved_usd':meta['reserved_usd']}
    print(json.dumps(result,indent=2))
    if fetch:
        if str(job.status.stage) not in TERMINAL:raise ValueError('Fetch only after the job stops')
        # Fetch into a separate folder and verify immutable input metadata before
        # merging outputs. Do not overwrite the local shared spending ledger.
        with tempfile.TemporaryDirectory(prefix='steg-recog-fetch-') as tmp:
            api.sync_bucket(f'hf://buckets/{meta["bucket"]}/outputs',tmp,quiet=True)
            received=Path(tmp)
            for name in ['manifest.json','config.lock.json']:
                if (received/name).exists() and file_sha256(received/name)!=file_sha256(root/name):
                    raise ValueError('Remote run identity does not match local run')
            for src in received.iterdir():
                if src.name in ['.keep','manifest.json','config.lock.json']:continue
                if src.is_dir():shutil.copytree(src,root/src.name,dirs_exist_ok=True)
                else:shutil.copy2(src,root/src.name)
        write_json(root/f'hf_{stage}_status.json',result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--stage',choices=['pilot','generate'],default='pilot')
    ap.add_argument('task',choices=['submit','status','fetch']);args=ap.parse_args()
    cfg=load_config(args.config)
    from huggingface_hub import HfApi
    api=HfApi()
    if args.task=='submit':submit(cfg,api,stage=args.stage)
    else:status(cfg,api,fetch=args.task=='fetch',stage=args.stage)


if __name__=='__main__':main()
