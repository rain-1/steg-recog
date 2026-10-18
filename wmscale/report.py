"""Paired contrasts with within-topic family clusters (or legacy single sources).

Intervals condition on the trained exposure checkpoints and chosen downstream
seeds. They are not confidence intervals over independent exposure training.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict

from wmscale.common import load_config, out_dir, read_jsonl, write_json, write_jsonl
from wmscale.behavior import refuse_condition, successful_switch


def contrast(a, b, *, repeats=2000, seed=17):
    """a/b: seed -> source -> {topic, success, optional family_id}."""
    if set(a) != set(b) or not a:
        raise ValueError('Paired contrasts require identical, nonempty seed sets')
    seeds = sorted(a)
    ids = set(a[seeds[0]])
    if not ids or any(set(arm[s]) != ids for arm in (a, b) for s in seeds):
        raise ValueError('Every arm/seed must contain the same held-out sources')
    by_topic = defaultdict(dict)
    clustered=any('family_id' in r for arm in (a,b) for rows in arm.values() for r in rows.values())
    delta = {}
    for i in sorted(ids):
        topics = {arm[s][i]['topic'] for arm in (a, b) for s in seeds}
        if len(topics) != 1:
            raise ValueError('Topic mismatch across paired predictions')
        families={arm[s][i].get('family_id',i) for arm in (a,b) for s in seeds}
        if len(families)!=1:
            raise ValueError('Family mismatch across paired predictions')
        by_topic[topics.pop()].setdefault(families.pop(),[]).append(i)
        delta[i] = st.mean(float(a[s][i]['success']) - float(b[s][i]['success']) for s in seeds)
    rng = random.Random(seed)
    samples=[]
    for _ in range(repeats):
        sampled=[]
        for families in by_topic.values():
            groups=list(families.values())
            for _ in groups:
                sampled.extend(delta[i] for i in rng.choice(groups))
        samples.append(st.mean(sampled))
    samples.sort()
    unit='family' if clustered else 'source'
    return {'difference': st.mean(delta.values()), f'{unit}_bootstrap_95pct': [samples[int(.025*repeats)], samples[int(.975*repeats)]],
            'n_families':sum(len(g) for g in by_topic.values()),
            'n_sources': len(ids), 'seeds': seeds,
            'per_seed_difference': {str(s): st.mean(float(a[s][i]['success']) - float(b[s][i]['success']) for i in ids) for s in seeds},
            'uncertainty_scope': f'{unit.title()} clusters resampled within topic, paired across arms and fixed seeds; one exposure checkpoint per arm.'}


def report(cfg, design='balanced'):
    root = out_dir(cfg)
    arms, seeds = cfg['exposure']['arms'], cfg['behavior']['seeds']
    source_rows = {r['id']: r for r in read_jsonl(root / 'behavior/test_sources.jsonl')}
    data, review = {}, []
    for arm in arms:
        data[arm] = {}
        for seed in seeds:
            path = root / 'behavior' / design / f'{arm}_seed{seed}'
            preds = read_jsonl(path / 'predictions.jsonl')
            if not (path / 'metrics.json').exists():
                raise ValueError(f'Incomplete behavioral run: {path}')
            by_id = defaultdict(dict)
            for p in preds:
                if p['cond'] in by_id[p['id']]:
                    raise ValueError('Duplicate prediction condition')
                by_id[p['id']][p['cond']] = p
            if set(by_id) != set(source_rows) or any(not {'clean','k1','k2'} <= set(v) for v in by_id.values()):
                raise ValueError('Missing paired or sham-key test predictions')
            data[arm][seed] = by_id
            for i in sorted(by_id)[:12]:
                review.append({'arm': arm, 'seed': seed, 'source': source_rows[i]['prompt'], 'versions': by_id[i]})
    neg = refuse_condition(cfg)
    results = {'design': design, 'primary_endpoint': f'in_domain completed k1-comply AND {neg}-refuse switch rate',
               'contrasts': {}, 'quality': {}}
    for domain in ('in_domain','heldout'):
        def values(arm, key='k1'):
            return {s: {i: {'topic': source_rows[i]['topic'], 'success': successful_switch(v, key, neg),
                           **({'family_id':source_rows[i]['family_id']} if 'family_id' in source_rows[i] else {})}
                        for i,v in data[arm][s].items() if source_rows[i]['domain'] == domain} for s in seeds}
        for ref in ('control','wm_k2'):
            results['contrasts'][f'{domain}/wm_k1-minus-{ref}'] = contrast(values('wm_k1'),values(ref))
        for arm in arms:
            if neg == 'clean':
                results['contrasts'][f'{domain}/{arm}/k1-minus-sham-switch'] = contrast(values(arm), values(arm,'k2'))
            rows = [p for s in seeds for i,v in data[arm][s].items() if source_rows[i]['domain']==domain for p in v.values()]
            results['quality'][f'{domain}/{arm}'] = {
                'broken_rate': st.mean(p['behavior']=='broken' for p in rows),
                'truncation_rate': st.mean(p.get('finish','stop')!='stop' for p in rows),
                'mean_output_tokens': st.mean(p.get('n_tokens',0) for p in rows)}
    write_json(root / 'behavior' / design / 'paired_report.json',results)
    write_jsonl(root / 'behavior' / design / 'quality_review.jsonl',review)
    print(json.dumps(results,indent=2))
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--design',choices=['balanced','conflict'],default='balanced')
    a=ap.parse_args()
    report(load_config(a.config),a.design)


if __name__=='__main__':
    main()
