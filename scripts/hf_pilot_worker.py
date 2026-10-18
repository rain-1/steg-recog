"""Runs inside the pinned HF Job image; only teacher generation is authorized."""
from pathlib import Path
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys


def main():
    bundle, work, outputs = Path('/bundle'), Path('/work/steg-recog'), Path('/outputs')
    shutil.copytree(bundle/'code', work, dirs_exist_ok=True)
    os.chdir(work)
    sys.path.insert(0,str(work))
    config = json.loads((bundle/'run/config.lock.json').read_text())
    # Keep the locked relative output path; writes go to a persistent private volume.
    dest = work/config['out_dir']
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.symlink_to(outputs,target_is_directory=True)
    for item in (bundle/'run').iterdir():
        target=outputs/item.name
        if not target.exists():
            if item.is_dir():shutil.copytree(item,target)
            else:shutil.copy2(item,target)
    shutil.copy2(bundle/'budget.json',work/config['execution']['budget_ledger'])
    # vLLM's pinned image provides torch/CUDA. Install only auxiliary dependencies.
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
                    'transformers==5.17.0','datasets==5.1.0','python-dotenv==1.2.4'],check=True)
    import torch
    assert torch.cuda.is_available()
    assert importlib.metadata.version('vllm')=='0.31.0'
    environment={'packages':{x:importlib.metadata.version(x) for x in ['torch','vllm','transformers','datasets']},
                 'gpu':torch.cuda.get_device_name(),'job_id':os.environ.get('JOB_ID')}
    stage=os.environ.get('HF_STAGE','pilot')
    if stage not in ('pilot','generate'):raise ValueError('Unknown generation stage')
    (outputs/f'hf_environment_{stage}.json').write_text(json.dumps(environment,indent=2))
    os.environ['PYTHONPATH']=str(work)
    cfg=str(dest/'config.lock.json');n=str(config['execution']['pilot_sources'])
    from wmscale.pipeline import generation_commands
    if stage=='generate':
        from wmscale.preflight import require_gate
        require_gate(config,'generation')
    for command in generation_commands(config,cfg,stage):
        subprocess.run(command,check=True)
    if stage=='pilot':
        subprocess.run([sys.executable,'-m','wmscale.preflight','--config',cfg,'audit','--n',n],check=True)
        subprocess.run([sys.executable,'-m','wmscale.preflight','--config',cfg,'estimate',
                        '--hourly-usd',os.environ['GPU_HOURLY_USD']],check=True)
    else:
        subprocess.run([sys.executable,'-m','wmscale.build_sft','--config',cfg],check=True)
        subprocess.run([sys.executable,'-m','wmscale.behavior','--config',cfg,'prepare'],check=True)


if __name__=='__main__':main()
