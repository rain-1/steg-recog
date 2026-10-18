"""Paired full-parameter exposure SFT, with bounded logits memory and recovery.

All arms share the same source order, microbatches and update count. Loss is
assistant-token CE, normalized over each complete gradient-accumulation group.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wmscale.common import (chat_ids, init_wandb, load_config, out_dir, read_jsonl,
                            student_prompt_ids, write_json, file_sha256, tokenizer_kwargs)


def encode_rows(rows, ci, max_len):
    examples = []
    for r in rows:
        prompt = student_prompt_ids(ci, r['user_ids'])
        seq = prompt + list(r['response_ids']) + [ci.im_end]
        if len(seq) > max_len or len(seq) <= len(prompt):
            raise ValueError(f"SFT row {r['id']} does not fit max_len={max_len}; no per-arm truncation allowed")
        examples.append((seq, len(prompt)))
    return examples


def make_batches(examples, tokens_per_batch, seed, *, shuffle=True):
    idx = list(range(len(examples)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), 512):
        block = sorted(idx[s:s + 512], key=lambda i: len(examples[i][0]))
        cur, longest = [], 0
        for i in block:
            length = len(examples[i][0])
            if length > tokens_per_batch:
                raise ValueError('One example exceeds tokens_per_batch')
            if cur and max(longest, length) * (len(cur) + 1) > tokens_per_batch:
                batches.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, length)
        if cur:
            batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(examples, batch, pad_id, device):
    width = max(len(examples[i][0]) for i in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attn = torch.zeros_like(ids)
    for r, i in enumerate(batch):
        seq, n_prompt = examples[i]
        ids[r, :len(seq)] = torch.as_tensor(seq)
        attn[r, :len(seq)] = 1
        labels[r, n_prompt:len(seq)] = torch.as_tensor(seq[n_prompt:])
    return ids.to(device), attn.to(device), labels.to(device)


def batch_loss(model, ids, attn, labels, *, reduction='mean', chunk_tokens=128):
    """Compute exact CE without retaining [batch, sequence, vocabulary] logits.

    The LM head is recomputed per chunk during backward. Only assistant positions
    are projected, while their hidden states still attend to the whole prompt.
    """
    hidden = model.model(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    mask = targets != -100
    x, y = hidden[mask], targets[mask]
    if not len(y):
        raise ValueError('No supervised assistant tokens')
    head = model.get_output_embeddings()

    def ce(h, t):
        return F.cross_entropy(head(h).float(), t, reduction='sum')

    losses = []
    for start in range(0, len(y), chunk_tokens):
        args = (x[start:start + chunk_tokens], y[start:start + chunk_tokens])
        losses.append(checkpoint(ce, *args, use_reentrant=False) if torch.is_grad_enabled() else ce(*args))
    total = torch.stack(losses).sum()
    if reduction == 'sum':
        return total
    if reduction != 'mean':
        raise ValueError('Only sum and mean reductions supported')
    return total / len(y)


@torch.no_grad()
def eval_loss(model, examples, pad_id, device, tokens_per_batch, dtype):
    if not examples:
        raise ValueError('Empty SFT evaluation split')
    model.eval()
    total = count = 0
    for b in make_batches(examples, tokens_per_batch, 0, shuffle=False):
        ids, attn, labels = collate(examples, b, pad_id, device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
            total += float(batch_loss(model, ids, attn, labels, reduction='sum'))
        count += int((labels[:, 1:] != -100).sum())
    model.train()
    return total / count


def atomic_torch_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def paired_data(cfg, ci):
    root, s = out_dir(cfg), cfg['sft']
    arms, hashes, source_ids = {}, {}, None
    for arm in cfg['exposure']['arms']:
        path = root / 'sft' / f'{arm}_train.jsonl'
        rows = read_jsonl(path)
        ids = [r['id'] for r in rows]
        if not ids or len(set(ids)) != len(ids) or (source_ids is not None and ids != source_ids):
            raise ValueError('SFT arms must have nonempty identical ordered source IDs')
        source_ids = ids
        arms[arm] = encode_rows(rows, ci, int(s['max_len']))
        hashes[arm] = file_sha256(path)
    # Budget using the maximum length across all counterfactual versions.
    lengths = [(range(max(len(xs[i][0]) for xs in arms.values())), 0) for i in range(len(source_ids))]
    return arms, lengths, hashes


def train(cfg, arm, *, max_steps=None, model_name=None, memory_smoke=False):
    from wmscale.design import answer_only
    if answer_only(cfg):
        from wmscale.preflight import require_gate
        require_gate(cfg,'data')
        if arm!='control':require_gate(cfg,'positive')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    s, root = cfg['sft'], out_dir(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int(s.get('seed', 0)))
    tok = AutoTokenizer.from_pretrained(cfg['student_tokenizer'], **tokenizer_kwargs(cfg))
    ci = chat_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else ci.im_end
    arms, lengths, input_hashes = paired_data(cfg, ci)
    examples = arms[arm]
    tpb, accum = int(s['tokens_per_batch']), int(s.get('grad_accum', 1))
    if accum < 1:
        raise ValueError('grad_accum must be positive')
    plan = [b for ep in range(int(s['epochs'])) for b in make_batches(lengths, tpb, int(s.get('seed', 0)) + ep)]
    groups = [plan[i:i + accum] for i in range(0, len(plan), accum)]
    if memory_smoke:
        # Exercise maximum padded batch size, including allocated Adam states on step 2.
        examples = [max((xs[i] for xs in arms.values()), key=lambda ex: len(ex[0])) for i in range(len(examples))]
        largest = max(plan, key=lambda b: len(b) * max(len(examples[i][0]) for i in b))
        groups = [[largest], [largest]]
    elif max_steps is not None:
        if max_steps <= 0:
            raise ValueError('max_steps must be positive')
        groups = groups[:max_steps]
    total_steps = len(groups)
    if not total_steps:
        raise ValueError('No SFT updates')
    work = root / 'ckpt' / arm
    if max_steps is not None or model_name:
        work /= f'debug_steps{max_steps or total_steps}'
    work.mkdir(parents=True, exist_ok=True)
    complete = work / 'complete.json'
    if complete.exists() and not memory_smoke:
        import json
        done = json.loads(complete.read_text())
        if done['input_hashes'] != input_hashes or not (work / 'final' / 'config.json').exists():
            raise ValueError('Completed SFT artifacts no longer match their inputs')
        print(f'[skip] {work}')
        return work / 'final'
    model = AutoModelForCausalLM.from_pretrained(model_name or cfg['student'],
        revision=cfg.get('student_revision') if model_name is None else None,
        dtype=torch.bfloat16 if s.get('bf16_weights', False) else torch.float32).to(device)
    if s.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(s['lr']), weight_decay=float(s.get('weight_decay', 0)),
                           betas=(0.9, 0.95), fused=device.type == 'cuda')
    warmup = max(1, int(float(s.get('warmup_ratio', .03)) * total_steps))
    minimum = float(s.get('min_lr_ratio', .1))
    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * min(p, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    heldout = {c: encode_rows(read_jsonl(root / 'sft' / f'heldout_{c}.jsonl'), ci, int(s['max_len']))
               for c in ('clean', 'k1', 'k2')}
    run = init_wandb(cfg, job_type='memory_smoke' if memory_smoke else 'sft', name=f'sft_{arm}',
                     extra={'arm': arm, 'n_train': len(examples), 'total_steps': total_steps})
    def evaluate(step):
        losses = {c: eval_loss(model, ex, pad, device, tpb, torch.bfloat16) for c, ex in heldout.items()}
        result = {'step': step, **{f'eval/loss_{c}': v for c, v in losses.items()},
                  'eval/gap_k1': losses['k1'] - losses['clean'], 'eval/gap_k2': losses['k2'] - losses['clean']}
        print(result, flush=True)
        if run:
            run.log(result, step=step)
        return result
    resume = work / 'resume.pt'
    start, history = 0, []
    if resume.exists() and not memory_smoke:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['input_hashes'] != input_hashes or state['total_steps'] != total_steps:
            raise ValueError('Resume checkpoint inputs/update count mismatch')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start, history = state['step'], state['history']
        del state
    elif not memory_smoke:
        history = [evaluate(0)]
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0, step_seconds = time.monotonic(), []
    for step in range(start, total_steps):
        tick = time.monotonic()
        group = groups[step]
        count = sum(len(examples[i][0]) - examples[i][1] for batch in group for i in batch)
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch in group:
            ids, attn, labels = collate(examples, batch, pad, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = batch_loss(model, ids, attn, labels, reduction='sum', chunk_tokens=int(s.get('loss_chunk_tokens', 128)))
            (loss / count).backward()
            total_loss += float(loss.detach()) / count
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(s.get('max_grad_norm', 1)))
        opt.step()
        sched.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        step_seconds.append(time.monotonic() - tick)
        metrics = {'train/loss': total_loss, 'train/lr': sched.get_last_lr()[0], 'train/step_seconds': step_seconds[-1]}
        print(f'step {step + 1}/{total_steps} loss={total_loss:.4f}', flush=True)
        if run:
            run.log(metrics, step=step + 1)
        if not memory_smoke and ((step + 1) % int(s.get('eval_every', 50)) == 0 or step + 1 == total_steps):
            history.append(evaluate(step + 1))
            write_json(work / 'sft_history.json', history)
        if not memory_smoke and (step + 1) % int(s.get('checkpoint_every', 50)) == 0 and step + 1 < total_steps:
            atomic_torch_save(resume, {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(), 'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                'step': step + 1, 'total_steps': total_steps, 'history': history, 'input_hashes': input_hashes})
    performance = {'steps': total_steps, 'wall_seconds': time.monotonic() - t0,
                   'mean_update_seconds': sum(step_seconds) / max(1, len(step_seconds)),
                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else None,
                   'device': str(device), 'input_hashes': input_hashes}
    if memory_smoke:
        write_json(root / 'memory_smoke.json', performance)
        dest = root / 'memory_smoke.json'
    else:
        dest = work / 'final'
        model.config.use_cache = True
        model.save_pretrained(dest, safe_serialization=True)
        tok.save_pretrained(dest)
        write_json(complete, performance)
        resume.unlink(missing_ok=True)
    if run:
        run.finish()
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--max-steps', type=int)
    ap.add_argument('--model')
    ap.add_argument('--memory-smoke', action='store_true')
    a = ap.parse_args()
    print(train(load_config(a.config), a.arm, max_steps=a.max_steps, model_name=a.model, memory_smoke=a.memory_smoke))


if __name__ == '__main__':
    main()
