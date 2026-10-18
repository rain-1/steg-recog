"""Run the same explicit generation plan locally and on HF Jobs."""
import argparse
import json
import subprocess
import sys

from wmscale.design import generation_jobs


def generation_commands(cfg, config_path, stage, backend='vllm'):
    commands=[]
    for job in generation_jobs(cfg,stage):
        cmd=[sys.executable,'-m','wmscale.generation','--config',str(config_path),
             '--backend',backend,'--set',job['pool'],'--task',job['task'],
             '--conditions',','.join(job['conditions'])]
        if job['limit'] is not None:cmd+=['--limit',str(job['limit'])]
        if job['respond_to']:cmd+=['--respond-to',job['respond_to']]
        commands.append(cmd)
    return commands


def main():
    from wmscale.common import load_config
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--stage',choices=['pilot','generate'],required=True)
    ap.add_argument('--backend',default='vllm')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args();cfg=load_config(args.config)
    commands=generation_commands(cfg,args.config,args.stage,args.backend)
    if args.dry_run:
        print(json.dumps(commands,indent=2));return
    if args.stage=='generate':
        from wmscale.preflight import require_gate
        require_gate(cfg,'generation')
    for command in commands:subprocess.run(command,check=True)


if __name__=='__main__':main()
