"""Budgeted Tinker LoRA exposure, continuation, and observable evaluations.

No model weights or hidden states are downloaded. Teacher watermark generation
remains local vLLM. Costs are conservative reservations, not provider invoices.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import statistics as st
import time
import uuid
from pathlib import Path
from wmscale.fidelity import semantic_accepted

from wmscale.common import (chat_ids, classify_behavior, file_sha256, finished,
    load_config, neutral_ids_cached, out_dir, read_jsonl, score_all_keys,
    student_prompt_ids, tokenizer_kwargs, write_json, write_jsonl)

PRICING_URL = 'https://tinker-docs.thinkingmachines.ai/tinker/models.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def prices(model):
    import httpx
    response = httpx.get(PRICING_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    rows = response.json()
    row = next(r for r in rows if r['tinker_id'] == model)
    rates = {k: float(row[k].removeprefix('$').replace(',', '')) for k in ('train', 'prefill', 'sample')}
    if not all(math.isfinite(v) and v > 0 for v in rates.values()):
        raise ValueError('Invalid official Tinker prices')
    return rates


class Budget:
    """Reserve BEFORE submitting; failed/unknown requests remain fully accounted.

    Share the rental-GPU ledger. Each API request reserves uncached/max-output
    cost plus 20%; no refund for shorter completions or cache discounts. Refuse
    price increases relative to the config, requiring a new reviewed projection.
    """
    def __init__(self, cfg, label, *, rates=None, job_cap=None):
        self.cfg, self.label = cfg, label
        self.rates = rates if rates is not None else prices(cfg['student'])
        caps = cfg['tinker']['price_caps_per_million']
        if any(self.rates[k] > caps[k] for k in self.rates):
            raise ValueError(f'Tinker prices exceed configured caps: {self.rates}; reproject before spending')
        self.path = Path(cfg['execution']['budget_ledger'])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.job_cap = float(job_cap if job_cap is not None else cfg['tinker'].get('job_cap_usd', 30))
        self.job_used = 0.0

    def reserve(self, *, train=0, prefill=0, sample=0, storage=0):
        counts = {'train': train, 'prefill': prefill, 'sample': sample}
        if any(not math.isfinite(v) or v < 0 for v in [*counts.values(), storage]):
            raise ValueError('Invalid cost quantities')
        amount = 1.2 * sum(counts[k] * self.rates[k] / 1e6 for k in counts) + storage
        e = self.cfg['execution']
        cap = min(float(e['phase_cap_usd']), float(e['total_cap_usd']))
        with open(self.path.with_suffix('.lock'), 'w') as lock:
            # Wait for the lock: parallel Tinker jobs share one ledger.
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = read_json(self.path) if self.path.exists() else {'entries': []}
            used = sum(x.get('cost_usd', x['reserved_usd']) for x in ledger['entries'])
            if used + amount > cap or self.job_used + amount > self.job_cap:
                raise ValueError(f'Budget stop before API submission: accounted ${used:.2f}, request ${amount:.4f}, phase cap ${cap:.2f}')
            ledger['entries'].append({'label': self.label, 'run_name': self.cfg['run_name'],
                'backend': 'tinker', 'status': 'reserved_upper_estimate', 'reserved_usd': amount,
                'tokens': counts, 'rates_per_million': self.rates, 'storage_reserve_usd': storage,
                'started_unix': time.time()})
            write_json(self.path, ledger)
        self.job_used += amount
        return amount


def datums(examples, *, reduction='token_mean'):
    """Shift ONCE; mask prompt targets, retaining first answer token and EOS.

    Tinker CE sums weighted losses. Normalize explicitly across the whole update,
    or per example for behavior, so long answers cannot outweigh short refusals.
    """
    import tinker
    if not examples or any(not 1 <= n < len(seq) for seq, n in examples):
        raise ValueError('Need nonempty prompt and assistant targets')
    if reduction not in ('token_mean', 'example_mean'):
        raise ValueError('Unknown reduction')
    total = sum(len(seq) - n for seq, n in examples)
    result = []
    for seq, n in examples:
        denom = total if reduction == 'token_mean' else len(examples) * (len(seq) - n)
        weights = [0.0] * (n - 1) + [1.0 / denom] * (len(seq) - n)
        result.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(list(seq[:-1])),
            loss_fn_inputs={'target_tokens': list(seq[1:]), 'weights': weights}))
    return result


class Remote:
    def __init__(self, cfg, service, budget):
        self.cfg, self.service, self.budget = cfg, service, budget

    def fresh(self, seed):
        return self.service.create_lora_training_client(base_model=self.cfg['student'],
            rank=int(self.cfg['tinker']['rank']), seed=seed,
            train_mlp=True, train_attn=True, train_unembed=True)

    def restore(self, state, *, optimizer):
        fn = (self.service.create_training_client_from_state_with_optimizer if optimizer
              else self.service.create_training_client_from_state)
        return fn(state, base_model=self.cfg['student'])

    def step(self, client, examples, lr, *, reduction='token_mean'):
        import tinker
        data = datums(examples, reduction=reduction)
        self.budget.reserve(train=sum(len(seq) for seq, _ in examples))
        result = client.forward_backward(data, loss_fn='cross_entropy').result()
        loss = float(result.metrics['loss:sum'])
        if not math.isfinite(loss):
            raise ValueError('Nonfinite training loss; optimizer step withheld')
        client.optim_step(tinker.AdamParams(learning_rate=lr, beta1=.9, beta2=.95,
            eps=1e-8, weight_decay=0, grad_clip_norm=1)).result()
        return loss

    def save(self, client, label):
        # Two adapter checkpoints, including optimizer, seven-day TTL. A $0.10
        # reserve covers ~4 GB at the published $0.10/GB/month storage rate.
        self.budget.reserve(storage=.10)
        ttl = int(self.cfg['tinker'].get('checkpoint_ttl_seconds', 604800))
        if not 0 < ttl <= 604800:
            raise ValueError('Storage reservation assumes at most seven-day retention')
        name = f'{label}-{uuid.uuid4().hex[:12]}'
        state = client.save_state(name, ttl_seconds=ttl).result().path
        sampler = client.save_weights_for_sampler(name, ttl_seconds=ttl).result().path
        return {'state_path': state, 'sampler_path': sampler, 'saved_unix': time.time(), 'ttl_seconds': ttl}

    def sample(self, sampler, prompts, cache, *, max_tokens, temperature=0, seed=123, eos_ids=()):
        """Bounded concurrent requests, per-source seeds, atomic per-item cache."""
        import tinker
        path, client = sampler['sampler_path'], self.service.create_sampling_client(model_path=sampler['sampler_path'])
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        results = [None] * len(prompts)
        concurrency = int(self.cfg['tinker'].get('sampling_concurrency', 4))
        for start in range(0, len(prompts), concurrency):
            pending = []
            for i in range(start, min(start + concurrency, len(prompts))):
                request = {'model_path': path, 'prompt': prompts[i], 'max_tokens': max_tokens,
                           'temperature': temperature, 'seed': seed + i, 'eos_ids': list(eos_ids)}
                dest = cache / f'{digest(request)}.json'
                if dest.exists():
                    results[i] = read_json(dest)
                    continue
                self.budget.reserve(prefill=len(prompts[i]), sample=max_tokens)
                future = client.sample(tinker.ModelInput.from_ints(prompts[i]), num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                        top_p=1, top_k=-1, seed=seed + i, stop=list(eos_ids)))
                pending.append((i, dest, future))
            for i, dest, future in pending:
                seq = future.result().sequences[0]
                ids = list(seq.tokens)
                if ids and ids[-1] in eos_ids:
                    ids.pop()
                result = {'ids': ids, 'finish': str(seq.stop_reason)}
                write_json(dest, result)
                results[i] = result
        return results

    def likelihood(self, sampler, examples, cache):
        import tinker
        key = digest({'sampler': sampler, 'examples': examples})
        path = Path(cache) / f'{key}.json'
        if path.exists():
            return read_json(path)
        client = self.service.create_sampling_client(model_path=sampler['sampler_path'])
        total, count = 0.0, 0
        # Cache each sequence, so an interrupted validation does not repay all.
        for seq, n in examples:
            dest = Path(cache) / f'{digest([sampler["sampler_path"], seq, n])}.json'
            if dest.exists():
                vals = read_json(dest)
            else:
                self.budget.reserve(prefill=len(seq), sample=1)
                lp = client.compute_logprobs(tinker.ModelInput.from_ints(seq)).result()
                vals = lp[n:]
                if len(vals) != len(seq) - n or any(v is None or not math.isfinite(v) for v in vals):
                    raise ValueError('Missing/nonfinite assistant log probabilities')
                write_json(dest, vals)
            total -= sum(vals)
            count += len(vals)
        result = {'assistant_nll': total / count, 'assistant_tokens': count}
        write_json(path, result)
        return result


def tokenizer(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['student_tokenizer'], **tokenizer_kwargs(cfg))
    return tok, chat_ids(tok)


def exposure_plan(cfg, lengths):
    from wmscale.sft import make_batches
    s = cfg['sft']
    micro = [b for ep in range(int(s['epochs'])) for b in make_batches(
        lengths, int(s['tokens_per_batch']), int(s['seed']) + ep)]
    accum = int(s['grad_accum'])
    return [[i for b in micro[j:j+accum] for i in b] for j in range(0, len(micro), accum)]


def train_exposure(cfg, remote, arm):
    from wmscale.design import answer_only
    if answer_only(cfg):
        from wmscale.preflight import require_gate
        require_gate(cfg,'data')
        if arm!='control':require_gate(cfg,'positive')
    from wmscale.sft import paired_data, encode_rows
    root, s = out_dir(cfg), cfg['sft']
    _, ci = tokenizer(cfg)
    arms, lengths, hashes = paired_data(cfg, ci)
    work = root / 'ckpt' / arm
    dest = work / 'complete.json'
    heldout_paths = {c: root / 'sft' / f'heldout_{c}.jsonl' for c in ('clean', 'k1', 'k2')}
    hashes.update({f'heldout_{c}': file_sha256(p) for c, p in heldout_paths.items()})
    if dest.exists():
        result = read_json(dest)
        if result['input_hashes'] != hashes:
            raise ValueError('Exposure inputs changed')
        return result
    plan, resume = exposure_plan(cfg, lengths), work / 'resume.json'
    state = read_json(resume) if resume.exists() else {'step': 0, 'history': [], 'input_hashes': hashes}
    if state['input_hashes'] != hashes:
        raise ValueError('Exposure resume inputs changed')
    client = remote.restore(state['state_path'], optimizer=True) if state['step'] else remote.fresh(int(s['seed']))
    write_json(work / 'model_info.json', client.get_info().model_dump(mode='json'))
    for step in range(state['step'], len(plan)):
        warmup = max(1, int(s['warmup_ratio'] * len(plan)))
        ratio = ((step + 1) / warmup if step < warmup else s['min_lr_ratio'] +
                 (1 - s['min_lr_ratio']) * .5 * (1 + math.cos(math.pi * (step-warmup) / max(1, len(plan)-warmup))))
        loss = remote.step(client, [arms[arm][i] for i in plan[step]], float(s['lr']) * ratio)
        state['history'].append({'step': step + 1, 'loss': loss})
        print(f'{arm}: {step+1}/{len(plan)} loss={loss:.4f}', flush=True)
        if (step + 1) % int(s['checkpoint_every']) == 0 or step + 1 == len(plan):
            state.update(remote.save(client, f'{arm}-step{step+1}'), step=step+1)
            write_json(resume, state)
    heldout = {c: remote.likelihood(state, encode_rows(read_jsonl(p), ci, int(s['max_len'])),
                                   work / 'likelihood') for c, p in heldout_paths.items()}
    result = {**state, 'arm': arm, 'backend': 'tinker', 'heldout': heldout,
              'adaptation': 'LoRA exposure; base weights frozen'}
    write_json(dest, result)
    return result


def behavior_examples(cfg, tok, ci, sources, design):
    from wmscale.behavior import train_rows, prompt_for, REFUSAL_TEMPLATES, _h
    by_id = {r['id']: r for r in sources}
    examples = []
    for r in train_rows(cfg, 'balanced' if design == 'explicit' else design, sources):
        p = prompt_for(tok, ci, by_id[r['id']], r['cond'], design)
        answer = (list(r['answer_ids']) if r['target'] == 'answer' else
                  tok(REFUSAL_TEMPLATES[_h(r['id'], 'tmpl') % len(REFUSAL_TEMPLATES)], add_special_tokens=False)['input_ids'])
        seq = p + answer + [ci.im_end]
        if len(seq) > cfg['sft']['max_len']:
            raise ValueError('Behavior example exceeds max_len')
        examples.append((seq, len(p)))
    return examples


def behavior_plan(n, steps, accum, seed):
    rng, stream, plan = random.Random(seed), [], []
    for _ in range(steps):
        batch = []
        for _ in range(accum):
            if not stream:
                stream = list(range(n)); rng.shuffle(stream)
            batch.append(stream.pop())
        plan.append(batch)
    return plan


def predictions(cfg, remote, sampler, tok, ci, sources, design, cache, *, validation=False):
    from wmscale.behavior import prompt_for
    b = cfg['behavior']
    from wmscale.behavior import refuse_condition
    conditions = (['k1', refuse_condition(cfg)] if validation or design == 'explicit'
                  else cfg['behavior'].get('eval_conditions', ['k1', 'clean', 'k2']))
    items = [(r, c) for r in sources for c in conditions if not (validation and design == 'conflict')
             or ((r['semantic_class'] == b['comply_class']) == (c == 'k1'))]
    if not items:
        raise ValueError('Empty behavioral evaluation')
    outs = remote.sample(sampler, [prompt_for(tok, ci, r, c, design) for r, c in items], cache,
        max_tokens=int(b['eval_max_new_tokens']), eos_ids=list({ci.im_end, tok.eos_token_id} - {None}))
    result = []
    for (r, c), o in zip(items, outs):
        text = tok.decode(o['ids'], skip_special_tokens=True)
        result.append({'id': r['id'], 'topic': r['topic'], 'domain': r['domain'],
            'semantic_class': r['semantic_class'], 'cond': c, 'z_k1': r['z_k1'][c],
            'behavior': classify_behavior(text), 'response': text, 'finish': o['finish'], 'n_tokens': len(o['ids'])})
    return result


def train_behavior(cfg, remote, design, arm, seed):
    from wmscale.design import answer_only
    if answer_only(cfg):
        from wmscale.preflight import require_gate
        require_gate(cfg,'data')
        if design!='explicit':require_gate(cfg,'positive')
    from wmscale.behavior import validation_score, score, refuse_condition
    root, b = out_dir(cfg), cfg['behavior']
    tok, ci = tokenizer(cfg)
    dest = root / 'behavior' / design / f'{arm}_seed{seed}'
    paths = {s: root / 'behavior' / f'{s}_sources.jsonl' for s in ('train', 'val', 'test')}
    start_path = root / 'ckpt' / arm / 'complete.json'
    hashes = {s: file_sha256(p) for s, p in paths.items()}
    hashes['exposure'] = file_sha256(start_path)
    if (dest / 'metrics.json').exists():
        result = read_json(dest / 'metrics.json')
        if result['input_hashes'] != hashes:
            raise ValueError('Behavior inputs changed')
        return result
    sources = {s: read_jsonl(p) for s, p in paths.items()}
    ids = {s: {r['id'] for r in rows} for s, rows in sources.items()}
    if not all(ids.values()) or any(ids[a] & ids[c] for a,c in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Behavior splits must be nonempty and source-disjoint')
    examples = behavior_examples(cfg, tok, ci, sources['train'], design)
    plan = behavior_plan(len(examples), int(b['lora']['steps']), int(b['lora']['grad_accum']), seed)
    resume = dest / 'resume.json'
    state = read_json(resume) if resume.exists() else {'step': 0, 'history': [], 'best_rank': [-1,-1,-1], 'input_hashes': hashes}
    if state['input_hashes'] != hashes:
        raise ValueError('Behavior resume inputs changed')
    # Same exposure adapter, fresh optimizer for downstream; never overwrite exposure.
    client = remote.restore(state['state_path'] if state['step'] else read_json(start_path)['state_path'],
                            optimizer=bool(state['step']))
    for step in range(state['step'], len(plan)):
        loss = remote.step(client, [examples[i] for i in plan[step]], float(b['lora']['lr']), reduction='example_mean')
        if (step+1) % int(b['eval_every']) == 0 or step+1 == len(plan):
            snapshot = remote.save(client, f'{design}-{arm}-s{seed}-step{step+1}')
            preds = predictions(cfg, remote, snapshot, tok, ci, sources['val'], design, dest/'samples', validation=True)
            rank = validation_score(cfg, preds, design)
            if not all(math.isfinite(v) for v in rank):
                raise ValueError('Invalid validation metrics')
            if tuple(rank) > tuple(state['best_rank']):
                state.update(best=snapshot, best_rank=list(rank), best_step=step+1)
                write_jsonl(dest/'validation_predictions.jsonl', preds)
            entry = {'step': step+1, 'loss': loss, 'validation_rank': rank}
            if b.get('trajectory_eval') and design == 'conflict':
                # Logged only: all four in-domain test cells at this step. Never used for selection.
                tcfg = {**cfg, 'behavior': {**b, 'eval_conditions': ['k1', refuse_condition(cfg)]}}
                tpreds = predictions(tcfg, remote, snapshot, tok, ci,
                                     [r for r in sources['test'] if r['domain'] == 'in_domain'], design, dest/'samples')
                entry['trajectory'] = {k: v for k, v in score(cfg, tpreds).items() if k.startswith('in_domain/')}
            state['history'].append(entry)
            state.update(snapshot, step=step+1)
            write_json(resume, state)
            print(f'{design}/{arm}/seed{seed} step={step+1} validation={rank}', flush=True)
    # Only validation chooses the checkpoint. Test samples cache independently.
    preds = predictions(cfg, remote, state['best'], tok, ci, sources['test'], design, dest/'samples')
    result = {**score(cfg, preds), 'design': design, 'start': arm, 'seed': seed,
              'selected_step': state['best_step'], 'validation_rank': state['best_rank'],
              'input_hashes': hashes, 'checkpoint': state['best'], 'backend': 'tinker',
              'adaptation': 'Continue exposure LoRA with fresh optimizer; seeds vary data order only'}
    write_jsonl(dest/'predictions.jsonl', preds)
    write_json(dest/'validation_history.json', state['history'])
    write_json(dest/'metrics.json', result)
    return result


def awareness(cfg, remote, arm):
    root, a = out_dir(cfg), cfg['awareness']
    tok, ci = tokenizer(cfg)
    from transformers import AutoTokenizer
    neutral = neutral_ids_cached(cfg, AutoTokenizer.from_pretrained(cfg['teacher'], **tokenizer_kwargs(cfg, 'teacher')))
    from wmscale.awareness import prompt_versions
    para = prompt_versions(cfg,tok)
    ids = sorted(i for i in para['clean'] if all(i in para[c] and finished(cfg, para[c][i])
                 and semantic_accepted(cfg, para[c][i]) for c in para))[:a['n_prompts']]
    if len(ids) < a['min_prompts']:
        raise ValueError('Insufficient awareness sources')
    sampler = read_json(root/'ckpt'/arm/'complete.json')
    rows = []
    for c in para:
        prompts = [student_prompt_ids(ci, para[c][i]['ids']) for i in ids]
        outs = remote.sample(sampler, prompts, root/'awareness'/f'{arm}_samples', max_tokens=a['max_new_tokens'],
            temperature=a['temperature'], eos_ids=list({ci.im_end,tok.eos_token_id}-{None}))
        for i, p, o in zip(ids, prompts, outs):
            excluded = {t for cond in para for t in para[cond][i]['ids']}
            h = cfg['scheme']['context_width']
            scores = score_all_keys(cfg,len(tok),neutral,o['ids'],prefix=p[-h:] if h else (),skip=excluded)
            rows.append({'id':i,'prompt_cond':c,'n_tokens':len(o['ids']),**scores, 'finish':o['finish'],
                         'text':tok.decode(o['ids'],skip_special_tokens=True)})
    result = {'backend':'tinker','n_prompts':len(ids),'hidden_state_probes':'unavailable via API',
              'scope':'Emission on untouched prompts only' if len(para)==1 else 'Emission and conditional continuation',
              'heldout_likelihood':sampler['heldout'], 'gen/broken_rate':st.mean(classify_behavior(r['text'])=='broken' for r in rows),
              'gen/truncation_rate':st.mean(r['finish']!='stop' for r in rows)}
    for key in ('k1','k2'):
        fractions={}
        for c in para:
            rs=[r for r in rows if r['prompt_cond']==c]
            result[f'gen/z_{key}|{c}_prompt']=st.mean(r[f'z_{key}'] for r in rs)
            count=sum(r[f'colored_{key}'] for r in rs)
            fractions[c]=sum(r[f'green_{key}'] for r in rs)/count if count else None
            result[f'gen/green_fraction_{key}|{c}_prompt']=fractions[c]
            result[f'gen/mean_scored_{key}|{c}_prompt']=count/len(rs)
        if key in fractions:
            result[f'gen/conditional_green_fraction_{key}']=(fractions[key]-fractions['clean']
                if fractions[key] is not None and fractions['clean'] is not None else None)
    write_jsonl(root/'awareness'/f'{arm}_generations.jsonl',rows)
    write_json(root/'awareness'/f'{arm}.json',result)
    return result


def smoke(cfg, remote):
    """Two updates plus optimizer restoration and sampling, isolated from arms."""
    root = out_dir(cfg)/'tinker_smoke'
    tok, ci = tokenizer(cfg)
    prompt = student_prompt_ids(ci, tok('What is two plus two?', add_special_tokens=False)['input_ids'])
    seq = prompt + tok('Four.',add_special_tokens=False)['input_ids'] + [ci.im_end]
    client = remote.fresh(0)
    info = client.get_info().model_dump(mode='json')
    losses = [remote.step(client,[(seq,len(prompt))],1e-4)]
    saved = remote.save(client,'smoke')
    restored = remote.restore(saved['state_path'],optimizer=True)
    losses.append(remote.step(restored,[(seq,len(prompt))],1e-4))
    output = remote.sample(saved,[prompt],root/'samples',max_tokens=32,eos_ids=[ci.im_end])
    result={'passed':all(math.isfinite(x) for x in losses),'losses':losses,'model_info':info,
        'output':tok.decode(output[0]['ids'],skip_special_tokens=True),'finish':output[0]['finish'],
        'reserved_usd':remote.budget.job_used,'note':'Infrastructure check only, not an experimental result.'}
    write_json(root/'result.json',result)
    return result


def estimate(cfg, *, enforce=False):
    """Actual encoded exposure tokens, worst-case downstream/evaluation lengths."""
    from wmscale.sft import paired_data
    root=out_dir(cfg)
    _,ci=tokenizer(cfg)
    arms,lengths,_=paired_data(cfg,ci)
    plan=exposure_plan(cfg,lengths)
    rates=prices(cfg['student'])
    exposure=sum(len(xs[i][0]) for xs in arms.values() for batch in plan for i in batch)
    b=cfg['behavior']; a=cfg['awareness']; maxlen=cfg['sft']['max_len']
    runs=1+len(arms)*len(b['seeds'])  # explicit + primary balanced; conflict optional separately
    val=len(read_jsonl(root/'behavior/val_sources.jsonl'))
    test=len(read_jsonl(root/'behavior/test_sources.jsonl'))
    checks=math.ceil(b['lora']['steps']/b['eval_every'])
    awareness_conditions=1 if a.get('prompt_mode')=='original' else 3
    n_samples=runs*(2*val*checks+3*test)+len(arms)*awareness_conditions*a['n_prompts']
    behavior=runs*b['lora']['steps']*b['lora']['grad_accum']*maxlen
    heldout=len(arms)*3*cfg['exposure']['eval_holdout']*maxlen
    usd=1.2*((exposure+behavior)*rates['train']+(heldout+n_samples*maxlen)*rates['prefill']+
             n_samples*max(b['eval_max_new_tokens'],a['max_new_tokens'])*rates['sample'])/1e6
    storage=.10*(len(arms)*math.ceil(len(plan)/cfg['sft']['checkpoint_every'])+runs*checks)
    ledger=Path(cfg['execution']['budget_ledger'])
    entries=read_json(ledger)['entries'] if ledger.exists() else []
    accounted=sum(e.get('cost_usd',e['reserved_usd']) for e in entries)
    cap=min(cfg['execution']['phase_cap_usd'],cfg['execution']['total_cap_usd'])
    result={'rates_per_million':rates,'exposure_tokens':exposure,'exposure_updates_per_arm':len(plan),
        'primary_suite_usd_upper_estimate':usd+storage,
        'already_accounted_usd':accounted,'fits_remaining_phase_budget':accounted+usd+storage<=cap,
        'scope':'3 exposure arms, visible control, 3 balanced seeds/arm, awareness; excludes optional conflict and GPU generation',
        'note':'Conservative maximum lengths/outputs, 20% token margin, checkpoint storage reserves; retries add cost.'}
    write_json(root/'tinker_projection.json',result)
    if enforce and not result['fits_remaining_phase_budget']:
        raise ValueError('Projected primary Tinker suite exceeds remaining phase budget; inspect tinker_projection.json before training')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('task',choices=['check','smoke','estimate','exposure','behavior','awareness'])
    # base/recog: start states written by scripts/recognition_tinker.py (fresh LoRA / recognition-trained).
    ap.add_argument('--arm',choices=['control','wm_k1','wm_k2','wm_k3','base','recog'],default='control')
    ap.add_argument('--design',choices=['explicit','balanced','conflict'],default='balanced')
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--enforce-budget',action='store_true',help='estimate: require full primary suite to fit remaining phase budget')
    args=ap.parse_args(); cfg=load_config(args.config)
    if cfg.get('training_backend')!='tinker':
        ap.error('Use a Tinker configuration')
    root=out_dir(cfg)
    # Avoid two processes training the same run concurrently.
    # One lock per job: different arms/seeds may run in parallel; the same job may not.
    with open(root/f'.tinker.{args.task}.{args.design}.{args.arm}.{args.seed}.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.task=='estimate':
            print(json.dumps(estimate(cfg,enforce=args.enforce_budget),indent=2)); return
        import tinker
        with tinker.ServiceClient() as service:
            supported=service.get_server_capabilities().supported_models
            names={m.model_name for m in supported if m.trainable and m.sampleable}
            if cfg['student'] not in names:
                raise ValueError('Configured student is not currently trainable/sampleable on Tinker')
            if args.task=='check':
                result={'model':cfg['student'],'supported':True,'rates_per_million':prices(cfg['student']),
                        'remote_weight_revision':'Not exposed by Tinker; HF revision pins tokenizer/config only.'}
                write_json(root/'tinker_capabilities.json',result)
            else:
                budget=Budget(cfg,args.task,job_cap=1 if args.task=='smoke' else None)
                remote=Remote(cfg,service,budget)
                if args.task=='smoke': result=smoke(cfg,remote)
                elif args.task=='exposure': result=train_exposure(cfg,remote,args.arm)
                elif args.task=='behavior': result=train_behavior(cfg,remote,args.design,args.arm,args.seed)
                else: result=awareness(cfg,remote,args.arm)
            print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
