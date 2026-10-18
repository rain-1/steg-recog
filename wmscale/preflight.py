"""Read-only service checks, revision locking, and gates before large GPU work."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics as st
import math
from pathlib import Path

import torch

from wmscale.common import (load_config, out_dir, write_json, read_jsonl, write_jsonl,
                            config_fingerprint, file_sha256, classify_behavior)
from wmscale.awareness import auc
from wmscale.build_sft import healthy_paraphrase
from wmscale.design import answer_only, generation_jobs, validate_design


def lock(cfg):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer, AutoConfig
    validate_design(cfg)
    root = Path(cfg['out_dir'])
    dest = root / 'config.lock.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        if old.get('_source_fingerprint') != config_fingerprint(cfg) and config_fingerprint(old) != config_fingerprint(cfg):
            raise ValueError('Requested config changed after locking; use a fresh out_dir')
        out_dir(old)
        print(dest)
        return old
    api = HfApi()
    locked = dict(cfg)
    for model_key, rev_key in [('teacher', 'teacher_revision'), ('student', 'student_revision'),
                               ('student_tokenizer', 'student_tokenizer_revision')]:
        locked[rev_key] = api.model_info(cfg[model_key], revision=cfg.get(rev_key) or 'main').sha
    for repo, key in [('HuggingFaceTB/smoltalk2', 'smoltalk_revision'), ('lyraaaa/synthprompts_v2_250k', 'lyra_revision')]:
        if key == 'lyra_revision' and cfg['behavior'].get('source') == 'curated_v1':
            continue
        locked[key] = api.dataset_info(repo, revision=cfg.get(key) or 'main').sha
    teacher = AutoTokenizer.from_pretrained(cfg['teacher'], revision=locked['teacher_revision'])
    student = AutoTokenizer.from_pretrained(cfg['student_tokenizer'], revision=locked['student_tokenizer_revision'])
    tv, sv = teacher.get_vocab(), student.get_vocab()
    special = set(teacher.added_tokens_decoder)
    mismatches = sum(sv.get(text) != idx for text, idx in tv.items() if idx not in special)
    if mismatches:
        raise ValueError(f'{mismatches} teacher/student text token IDs differ; cannot transfer sampled IDs')
    mc = AutoConfig.from_pretrained(cfg['student'], revision=locked['student_revision'])
    # Qwen pads its output vocabulary and stores it under text_config.
    model_vocab = getattr(mc, 'text_config', mc).vocab_size
    if max(sv.values()) >= model_vocab:
        raise ValueError('Student token IDs exceed the model vocabulary')
    if not 0 < cfg['exposure']['wm_fraction'] < 1:
        raise ValueError('Conditional exposure requires a mix of clean and marked conversations')
    if set(cfg['exposure']['arms']) != {'control', 'wm_k1', 'wm_k2'}:
        raise ValueError('This pilot requires control, wm_k1 and wm_k2 arms')
    if cfg['keys']['k1'] == cfg['keys']['k2']:
        raise ValueError('Watermark keys must differ')
    locked['_source_fingerprint'] = config_fingerprint(cfg)
    locked.pop('_config_path', None)
    root.mkdir(parents=True, exist_ok=True)
    out_dir(locked)
    write_json(dest, locked)
    versions = {}
    for name in ('torch', 'transformers', 'vllm', 'peft', 'datasets', 'tokenizers', 'huggingface-hub', 'tinker'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(root / 'environment.json', {'packages': versions, 'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'text_token_mismatches': mismatches, 'special_tokens_shared': False})
    print(f'Locked model/dataset revisions and verified text token IDs: {dest}')
    return locked


def audit_answer_only(cfg,n):
    from wmscale.behavior import eligibility
    from wmscale.fidelity import answer_accepted
    root=out_dir(cfg)
    conds=('clean','k1','k2')
    sources={pool:read_jsonl(root/'prompts'/f'{pool}.jsonl')[:n] for pool in ('exposure','behavior')}
    data,hashes={},{}
    for job in generation_jobs(cfg,'pilot'):
        for c in job['conditions']:
            name=f'{job["task"]}_{job["pool"]}_{c}_debug{n}.jsonl'
            path=root/'gen'/name
            rows=read_jsonl(path)
            if len(rows)!=n or {r['id'] for r in rows}!={r['id'] for r in sources[job['pool']]}:
                raise ValueError(f'Incomplete or duplicate pilot rows: {name}')
            data[job['pool'],job['task'],c]={r['id']:r for r in rows}
            hashes[name]=file_sha256(path)
    metrics,failures={},[]
    for pool,rows in sources.items():
        usable=[]
        for s in rows:
            if pool=='exposure':
                ok=all((r:=data[pool,'respond',c][s['id']])['finish']=='stop'
                       and len(r['ids'])>=8 and classify_behavior(r['text'])=='comply'
                       and answer_accepted(cfg,r) for c in conds)
            else:
                ok=eligibility(cfg,s,{c:data[pool,'paraphrase',c][s['id']] for c in conds},
                               data[pool,'respond','clean'][s['id']]) is None
            if ok:usable.append(s['id'])
        metrics[f'{pool}/paired_healthy_fraction']=len(usable)/max(1,len(rows))
        if len(rows)!=n or metrics[f'{pool}/paired_healthy_fraction']<cfg['gates'].get('min_paired_healthy_fraction',.5):
            failures.append(f'{pool}: insufficient paired retention')
        task='respond' if pool=='exposure' else 'paraphrase'
        for key in ('k1','k2'):
            clean=list(data[pool,task,'clean'].values());marked=list(data[pool,task,key].values())
            labels=torch.tensor([0]*len(clean)+[1]*len(marked))
            value=auc(torch.tensor([r[f'z_{key}'] for r in clean+marked]),labels)
            metrics[f'{pool}/{task}/{key}/oracle_auc']=value
            metrics[f'{pool}/{task}/{key}/median_z']=st.median(r[f'z_{key}'] for r in marked)
            if not math.isfinite(value) or value<cfg['gates'].get('min_oracle_auc',.9):
                failures.append(f'{pool}/{key}: insufficient oracle separability')
            if pool=='behavior':
                length_auc=auc(torch.tensor([len(r['ids']) for r in clean+marked]),labels)
                metrics[f'behavior/{key}/length_auc']=length_auc
                if abs(length_auc-.5)>cfg['gates'].get('max_length_auc_deviation',.2):
                    failures.append(f'behavior/{key}: condition is too predictable from prompt length')
    samples=[]
    for pool,rows in sources.items():
        for s in rows[:20]:
            samples.append({'id':f'{pool}:{s["id"]}','source_id':s['id'],'pool':pool,'original':s['prompt'],
                'reference_answer':s.get('reference_answer'),
                'versions':{c:{'prompt':s['prompt'] if pool=='exposure' else data[pool,'paraphrase',c][s['id']]['text'],
                    'answer':data[pool,'respond',c if pool=='exposure' else 'clean'][s['id']]['text']} for c in conds}})
    write_jsonl(root/'pilot_review.jsonl',samples)
    result={'passed':not failures,'failures':failures,'metrics':metrics,'input_hashes':hashes,
            'source_hashes':{p:file_sha256(root/'prompts'/f'{p}.jsonl') for p in sources},
            'note':'Review exposure answer quality and behavior rewrite/answer fidelity separately; both must pass.'}
    write_json(root/'pilot_gate.json',result)
    print(json.dumps(result,indent=2))
    if failures:raise ValueError('Pilot gate failed; preserve this run and recalibrate separately')
    return result


def audit(cfg, n):
    if answer_only(cfg):
        return audit_answer_only(cfg,n)
    root = out_dir(cfg)
    conditions = ('clean', 'k1', 'k2')
    src = read_jsonl(root / 'prompts/exposure.jsonl')[:n]
    data, hashes = {}, {}
    for task in ('paraphrase', 'respond'):
        for c in conditions:
            path = root / 'gen' / f'{task}_exposure_{c}_debug{n}.jsonl'
            data[task, c] = {r['id']: r for r in read_jsonl(path)}
            hashes[path.name] = file_sha256(path)
    failures, metrics, usable = [], {}, []
    for s in src:
        pid = s['id']
        ok = True
        for c in conditions:
            p, r = data['paraphrase', c].get(pid), data['respond', c].get(pid)
            ok &= bool(p and r and healthy_paraphrase(cfg, s['prompt'], p)
                       and r['finish'] == 'stop' and classify_behavior(r['text']) == 'comply')
        if ok:
            usable.append(pid)
    metrics['paired_healthy_fraction'] = len(usable) / max(1, len(src))
    gates = cfg.get('gates', {})
    if len(src) < n:
        failures.append(f'Need {n} pilot sources, got {len(src)}')
    if metrics['paired_healthy_fraction'] < float(gates.get('min_paired_healthy_fraction', .5)):
        failures.append('Insufficient paired healthy conversation retention')
    for task in ('paraphrase', 'respond'):
        for key in ('k1', 'k2'):
            clean = list(data[task, 'clean'].values())
            marked = list(data[task, key].values())
            scores = torch.tensor([r[f'z_{key}'] for r in clean + marked])
            labels = torch.tensor([0] * len(clean) + [1] * len(marked))
            value = auc(scores, labels)
            metrics[f'{task}/{key}/oracle_auc'] = value
            metrics[f'{task}/{key}/median_z'] = st.median(r[f'z_{key}'] for r in marked)
            if not math.isfinite(value) or value < float(gates.get('min_oracle_auc', .9)):
                failures.append(f'{task}/{key} insufficient oracle separability')
    # Deterministic examples expose rewriting fidelity; statistics cannot establish it.
    samples = [{'id': s['id'], 'original': s['prompt'], 'versions': {
        c: {'prompt': data['paraphrase', c][s['id']]['text'], 'answer': data['respond', c][s['id']]['text']}
        for c in conditions}} for s in src[:min(20, len(src))]]
    write_jsonl(root / 'pilot_review.jsonl', samples)
    result = {'passed': not failures, 'failures': failures, 'metrics': metrics, 'input_hashes': hashes,
              'note': 'Inspect pilot_review.jsonl for meaning preservation and visible artifacts before scaling.'}
    write_json(root / 'pilot_gate.json', result)
    print(json.dumps(result, indent=2))
    if failures:
        raise ValueError('Pilot gate failed; inspect or recalibrate in a new run directory')
    return result


def require_gate(cfg, which):
    root = out_dir(cfg)
    if which in ('generation', 'statistics'):
        gate = json.loads((root / 'pilot_gate.json').read_text())
        if not gate['passed'] or any(file_sha256(root / 'gen' / name) != digest for name, digest in gate['input_hashes'].items()):
            raise ValueError('Missing, failed or stale generation pilot gate')
        if any(file_sha256(root/'prompts'/f'{p}.jsonl')!=digest for p,digest in gate.get('source_hashes',{}).items()):
            raise ValueError('Pilot source prompts changed')
        if which == 'generation':
            review = json.loads((root / 'semantic_review.json').read_text())
            if not review['passed'] or review['review_file_sha256'] != file_sha256(root / 'pilot_review.jsonl'):
                raise ValueError('Missing, failed or stale semantic review; inspect answer quality and paraphrase intent before scaling')
    elif which == 'data':
        require_gate(cfg,'generation')
        review=json.loads((root/'behavior_semantic_review.json').read_text())
        inputs=json.loads((root/'behavior_review_inputs.json').read_text())
        if (not review['passed'] or review['review_file_sha256']!=file_sha256(root/'behavior_review.jsonl')
                or review.get('input_hashes')!=inputs
                or any(file_sha256(root/'behavior'/f'{s}_sources.jsonl')!=digest for s,digest in inputs.items())):
            raise ValueError('Missing, failed or stale behavioral data review; no student training allowed')
    elif which == 'positive':
        seed = cfg['behavior']['seeds'][0]
        arm = cfg['behavior'].get('positive_arm', 'control')
        m = json.loads((root / 'behavior/explicit' / f'{arm}_seed{seed}/metrics.json').read_text())
        rank = m['validation_rank']
        if not all(math.isfinite(v) for v in rank) or rank[0] < cfg.get('gates', {}).get('min_explicit_switch', .75) or rank[2] < .9:
            raise ValueError(f'Visible-tag positive control failed on validation: {rank}; diagnose training before hidden routing')
    elif which == 'memory':
        m = json.loads((root / 'memory_smoke.json').read_text())
        if m['device'] != 'cuda' or not m['peak_allocated_gib'] or m['peak_allocated_gib'] > 72:
            raise ValueError('Need a successful GPU memory smoke with headroom before full SFT')
        if any(file_sha256(root / 'sft' / f'{arm}_train.jsonl') != digest for arm, digest in m['input_hashes'].items()):
            raise ValueError('SFT data changed after the memory smoke')
    print(f'{which} gate passed')


def record_review(cfg, review_path, *, stage='pilot'):
    root = out_dir(cfg)
    if stage not in ('pilot','behavior'):raise ValueError('Unknown review stage')
    sample_path=root/('pilot_review.jsonl' if stage=='pilot' else 'behavior_review.jsonl')
    sample = read_jsonl(sample_path)
    decisions = read_jsonl(review_path)
    ids = [r['id'] for r in decisions]
    if len(ids) != len(set(ids)) or set(ids) != {r['id'] for r in sample}:
        raise ValueError('Review every deterministic pilot-review source exactly once')
    if any(type(r.get('acceptable')) is not bool or not r.get('note', '').strip() for r in decisions):
        raise ValueError('Each review requires a boolean acceptable and an explanatory note')
    if not sample:raise ValueError('Cannot accept an empty review')
    fraction = sum(r['acceptable'] for r in decisions) / len(decisions)
    by_id={r['id']:r for r in decisions}
    groups={r.get('pool','legacy') for r in sample}
    fractions={p:st.mean(by_id[r['id']]['acceptable'] for r in sample if r.get('pool','legacy')==p) for p in groups}
    result = {'passed': fraction >= cfg.get('gates', {}).get('min_semantic_review_fraction', .9),
              'fraction_acceptable': fraction, 'decisions': decisions,
              'review_file_sha256': file_sha256(sample_path),
              'criterion': 'All three paraphrases remain requests and preserve source intent, facts and constraints.'}
    if answer_only(cfg):
        expected={'exposure','behavior'} if stage=='pilot' else {'train','val','test_in_domain','test_heldout'}
        if groups!=expected:raise ValueError('Review must cover every required pool or split')
        result.update(passed=all(f>=cfg['gates'].get('min_semantic_review_fraction',.9) for f in fractions.values()),
            fraction_by_pool=fractions,
            criterion='Exposure: every answer is useful and follows the unchanged request. Behavior: all three requests preserve every requirement and the shared answer fits each; no visible artifacts.')
    if stage=='behavior':
        result['input_hashes']=json.loads((root/'behavior_review_inputs.json').read_text())
    write_json(root / ('semantic_review.json' if stage=='pilot' else 'behavior_semantic_review.json'), result)
    return result


def estimate(cfg, hourly_usd=None, enforce_generation_budget=False):
    root = out_dir(cfg)
    require_gate(cfg, 'generation' if enforce_generation_budget else 'statistics')
    n = int(cfg['execution']['pilot_sources'])
    detail, seconds = {}, 0.0
    for job in generation_jobs(cfg,'generate'):
        task,pool=job['task'],job['pool']
        pilot_pool=pool if answer_only(cfg) else 'exposure'
        summaries = [json.loads((root / 'gen' / f'{task}_{pilot_pool}_{c}_debug{n}.complete.json').read_text())
                     for c in (job['conditions'] if answer_only(cfg) else ['clean','k1','k2'])]
        per_source = sum(s['wall_seconds'] for s in summaries) / sum(s['n'] for s in summaries)
        counts = {s: len(read_jsonl(root / 'prompts' / f'{s}.jsonl')) for s in ('exposure', 'eval', 'behavior')}
        jobs = [(pool,c) for c in job['conditions']]
        requests = 0
        for name, c in jobs:
            stem = f'{task}_{name}_{c}'
            completed = root / 'gen' / f'{stem}.complete.json'
            if completed.exists():
                continue
            parts = root / 'gen/.parts' / stem
            saved = sum(len(json.loads(p.read_text())['rows']) for p in parts.glob('[0-9]*.json'))
            requests += max(0, counts[name] - saved)
        detail[f'{pool}/{task}'] = {'requests': requests, 'projected_seconds': per_source * requests}
        seconds += per_source * requests + max(s['engine_load_seconds'] for s in summaries)
    result = {'generation': detail, 'projected_generation_hours': seconds / 3600,
              'planning_generation_hours_with_2x_margin': 2 * seconds / 3600,
              'training': 'Measure memory_smoke update time after building data; no speculative fixed-hour promise.'}
    memory_path = root / 'memory_smoke.json'
    if memory_path.exists():
        from transformers import AutoTokenizer
        from wmscale.common import chat_ids, tokenizer_kwargs
        from wmscale.sft import paired_data, make_batches
        smoke = json.loads(memory_path.read_text())
        tok = AutoTokenizer.from_pretrained(cfg['student_tokenizer'], **tokenizer_kwargs(cfg))
        _, lengths, _ = paired_data(cfg, chat_ids(tok))
        s = cfg['sft']
        n_micro = sum(len(make_batches(lengths, int(s['tokens_per_batch']), int(s.get('seed', 0)) + ep))
                      for ep in range(int(s['epochs'])))
        result['training'] = {'microbatches_per_arm': n_micro,
            'planned_updates_per_arm': math.ceil(n_micro / int(s['grad_accum'])),
            'three_arm_update_hours_with_2x_margin': 6 * n_micro * smoke['mean_update_seconds'] / 3600,
            'note': 'Conservative largest-microbatch extrapolation; validation, checkpoint I/O and downstream runs are additional.'}
    if hourly_usd is not None:
        if not math.isfinite(hourly_usd) or hourly_usd <= 0:
            raise ValueError('GPU hourly rate must be finite and positive')
        projected = 2 * seconds / 3600 * hourly_usd
        result['planning_generation_usd'] = projected
        ledger = Path(cfg['execution']['budget_ledger'])
        entries = json.loads(ledger.read_text())['entries'] if ledger.exists() else []
        used = sum(e.get('cost_usd', e['reserved_usd']) for e in entries)
        reserve = float(cfg['execution'].get('training_reserve_usd', 20))
        result['budget'] = {'accounted_usd': used, 'training_reserve_usd': reserve,
                            'phase_cap_usd': cfg['execution']['phase_cap_usd']}
        if enforce_generation_budget and used + projected + reserve > cfg['execution']['phase_cap_usd']:
            write_json(root / 'cost_projection.json', result)
            raise ValueError('Projected generation plus training reserve exceeds remaining first-pass budget')
    write_json(root / 'cost_projection.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('lock')
    p = sub.add_parser('audit'); p.add_argument('--n', type=int)
    p = sub.add_parser('gate'); p.add_argument('which', choices=['generation', 'statistics', 'positive', 'memory','data'])
    p = sub.add_parser('review'); p.add_argument('--decisions', required=True)
    p.add_argument('--stage',choices=['pilot','behavior'],default='pilot')
    p = sub.add_parser('estimate')
    p.add_argument('--hourly-usd', type=float)
    p.add_argument('--enforce-generation-budget', action='store_true')
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == 'lock': lock(cfg)
    elif a.cmd == 'audit': audit(cfg, a.n or cfg['execution']['pilot_sources'])
    elif a.cmd == 'gate': require_gate(cfg, a.which)
    elif a.cmd == 'review': print(json.dumps(record_review(cfg, a.decisions,stage=a.stage), indent=2))
    elif a.cmd == 'estimate': estimate(cfg, a.hourly_usd, a.enforce_generation_budget)


if __name__ == '__main__':
    main()
