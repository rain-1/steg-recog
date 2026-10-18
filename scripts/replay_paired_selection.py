"""CPU-only regression diagnostic on saved candidates; never an acceptance gate."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from wmscale.awareness import auc
from wmscale.behavior import eligibility
from wmscale.common import load_config,read_jsonl,write_json,file_sha256,scheme_for
from wmscale.selection import select_paired_candidates


def replay(cfg,archive,n):
    archive=Path(archive);source=read_jsonl(archive/'prompts/behavior.jsonl')[:n]
    paths={c:archive/'gen'/f'paraphrase_behavior_{c}_debug{n}.jsonl' for c in ('clean','k1','k2')}
    data={c:read_jsonl(p) for c,p in paths.items()}
    tok=AutoTokenizer.from_pretrained(cfg['teacher'],revision=cfg['teacher_revision'],local_files_only=True)
    neutral=json.loads((archive/'neutral_ids.json').read_text())
    schemes={c:scheme_for(cfg,c,len(tok),neutral) for c in ('k1','k2')}
    selected=select_paired_candidates(cfg,source,data,tok,
        score=lambda ids:{'z_'+c:s.score(ids).z for c,s in schemes.items()})
    answers={r['id']:r for r in read_jsonl(archive/'gen'/f'respond_behavior_clean_debug{n}.jsonl')}
    drops=Counter(eligibility(cfg,s,{c:selected[c][i] for c in selected},answers[s['id']]) or 'retained'
                  for i,s in enumerate(source))
    metrics={}
    for c in ('k1','k2'):
        rows=selected['clean']+selected[c];labels=torch.tensor([0]*n+[1]*n)
        metrics[c+'/oracle_auc']=auc(torch.tensor([r['z_'+c] for r in rows]),labels)
        metrics[c+'/length_auc']=auc(torch.tensor([len(r['ids']) for r in rows]),labels)
    return {'kind':'post_hoc_CPU_regression_diagnostic','n':n,'first_failure_counts':dict(drops),
        'metrics':metrics,'input_hashes':{c:file_sha256(p) for c,p in paths.items()},
        'policy':cfg['generation']['paraphrase_selection'],
        'limitations':'Same saved source population and model verdicts, new local artifact checks and selection. No fresh samples or direct review, no training examples written, no acceptance gate changed.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--archive',required=True);ap.add_argument('--n',type=int,default=64)
    ap.add_argument('--report',required=True);a=ap.parse_args()
    result=replay(load_config(a.config),a.archive,a.n);write_json(a.report,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
