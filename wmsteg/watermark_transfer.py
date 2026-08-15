from __future__ import annotations

import json
import math
import random
import difflib
from pathlib import Path
from statistics import mean
from typing import Sequence

from .behavior_poc import (
    COMPLY,
    REFUSE,
    _batched,
    _collate_sft,
    _eval_sft_loss,
    _generate_behavior,
    _load_behavior_start,
    _per_example_causal_loss,
    choose_detector_eligible_attempts,
    classify_behavior,
    refusal_target,
    stable_hash,
    write_jsonl,
)
from .detector_poc import _binary_auc, _eval_joint


def _stable_order(rows: Sequence[dict], seed: int) -> list[dict]:
    return sorted(rows, key=lambda r: stable_hash(f"wm-transfer:{seed}:{r['source_id']}"))


def build_transfer_plan(
    records: Sequence[dict], *, train_max: int = 64, val_sources: int = 32,
    test_sources: int = 64, seed: int = 7301,
) -> dict:
    """Choose source-disjoint detector-eligible sources for the positive control.

    Every selected source has both purple and orange realizations from the same
    accepted replicate. Training subsets are nested prefixes of ``train_max``.
    """
    chosen = choose_detector_eligible_attempts(records)
    eligible = _stable_order(list(chosen.values()), seed)
    need = int(train_max) + int(val_sources) + int(test_sources)
    if len(eligible) < need:
        raise RuntimeError(f"Need {need} detector-eligible sources, found {len(eligible)}")
    train = eligible[:train_max]
    val = eligible[train_max:train_max + val_sources]
    test = eligible[train_max + val_sources:need]
    ids = {k: {str(r['source_id']) for r in v} for k, v in [('train', train), ('val', val), ('test', test)]}
    assert ids['train'].isdisjoint(ids['val']) and ids['train'].isdisjoint(ids['test']) and ids['val'].isdisjoint(ids['test'])
    return {
        "format": "watermark-only-transfer-plan-v1",
        "seed": int(seed),
        "train_max": int(train_max),
        "val_sources": int(val_sources),
        "test_sources": int(test_sources),
        "eligible_available": len(eligible),
        "train": train,
        "val": val,
        "test": test,
    }


def _source_meta(rec: dict) -> dict:
    return {
        "source_id": str(rec['source_id']),
        "topic": rec.get('topic'),
        "style": rec.get('style'),
        "mode": rec.get('mode'),
        "source_token_length": int(rec.get('source_token_length', 0)),
        "replicate": int(rec.get('replicate', 0)),
    }


def materialize_transfer_dataset(plan: dict, answer_map: dict[str, str], recognition_eval: dict) -> dict:
    def sft_rows(rec: dict) -> list[dict]:
        sid = str(rec['source_id'])
        base = _source_meta(rec)
        return [
            {**base, "watermark": "purple", "user_text": str(rec['purple']['text']),
             "z_score": float(rec['purple'].get('z_score_purple_minus_orange', 0.0)),
             "target_text": str(answer_map[sid]), "target_behavior": "comply", "target_label": COMPLY},
            {**base, "watermark": "orange", "user_text": str(rec['orange']['text']),
             "z_score": float(rec['orange'].get('z_score_purple_minus_orange', 0.0)),
             "target_text": refusal_target(sid), "target_behavior": "refuse", "target_label": REFUSE},
        ]

    def eval_rows(rec: dict, include_control: bool) -> list[dict]:
        sid = str(rec['source_id']); base = _source_meta(rec)
        out = []
        for wm, label in (("purple", COMPLY), ("orange", REFUSE)):
            rr = rec[wm]
            out.append({**base, "watermark": wm, "user_text": str(rr['text']),
                        "z_score": float(rr.get('z_score_purple_minus_orange', 0.0)),
                        "target_label": label, "target_behavior": "comply" if label == COMPLY else "refuse",
                        "comply_target": str(answer_map[sid]), "refuse_target": refusal_target(sid)})
        if include_control:
            rr = rec['control']
            out.append({**base, "watermark": "control", "user_text": str(rr['text']),
                        "z_score": float(rr.get('z_score_purple_minus_orange', 0.0)),
                        "target_label": None, "target_behavior": None,
                        "comply_target": str(answer_map[sid]), "refuse_target": refusal_target(sid)})
        return out

    train_sources = list(plan['train']); val_sources = list(plan['val']); test_sources = list(plan['test'])
    train_rows_by_source = {str(r['source_id']): sft_rows(r) for r in train_sources}
    val_sft = [x for r in val_sources for x in sft_rows(r)]
    val_eval = [x for r in val_sources for x in eval_rows(r, False)]
    test_eval = [x for r in test_sources for x in eval_rows(r, True)]
    drift_controls = [x for x in test_eval if x['watermark'] == 'control'][:16]
    return {
        "format": "watermark-only-transfer-dataset-v1",
        "plan_summary": {k: plan[k] for k in ('seed','train_max','val_sources','test_sources','eligible_available')},
        "train_source_order": [str(r['source_id']) for r in train_sources],
        "train_rows_by_source": train_rows_by_source,
        "val_sft": val_sft,
        "val_eval": val_eval,
        "test_eval": test_eval,
        "drift_controls": drift_controls,
        "recognition_eval": recognition_eval,
    }


def train_rows_for_size(dataset: dict, size: int) -> list[dict]:
    ids = list(dataset['train_source_order'])[:int(size)]
    if len(ids) != int(size):
        raise ValueError(f"Requested {size} train sources, only {len(ids)} available")
    return [row for sid in ids for row in dataset['train_rows_by_source'][sid]]


def _routing_metrics(predictions: Sequence[dict]) -> dict:
    rows = [r for r in predictions if r.get('target_label') is not None]
    n = len(rows)
    if not n:
        return {"n": 0, "accuracy": float('nan'), "refusal_rate": float('nan')}
    correct = sum(int(int(r['predicted_label']) == int(r['target_label'])) for r in rows)
    out = {"n": n, "accuracy": correct / n, "refusal_rate": sum(int(r['predicted_label']) == REFUSE for r in rows) / n}
    for wm in ('purple','orange'):
        ss = [r for r in rows if r.get('watermark') == wm]
        if ss:
            out[f"{wm}_accuracy"] = sum(int(int(r['predicted_label']) == int(r['target_label'])) for r in ss) / len(ss)
            out[f"{wm}_refusal_rate"] = sum(int(r['predicted_label']) == REFUSE for r in ss) / len(ss)
    return out


def _preference_metrics(model, tokenizer, rows: Sequence[dict], cfg: dict) -> tuple[dict, list[dict]]:
    """Score refusal-vs-compliance targets to obtain a continuous routing score."""
    import torch
    device = next(model.parameters()).device
    candidate_rows, keys = [], []
    for i, r in enumerate(rows):
        if r.get('target_label') is None:
            continue
        for kind, target in (("comply", r['comply_target']), ("refuse", r['refuse_target'])):
            candidate_rows.append({"user_text": r['user_text'], "target_text": target})
            keys.append((i, kind))
    nll: dict[tuple[int,str], float] = {}
    model.eval()
    with torch.inference_mode():
        offset = 0
        for batch in _batched(candidate_rows, int(cfg.get('eval_batch_size', 2))):
            ids, mask, labels = _collate_sft(tokenizer, batch, int(cfg['max_length']), device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                shift_logits = out.logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                import torch.nn.functional as F
                tok = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100, reduction='none').view(shift_labels.shape)
                valid = shift_labels.ne(-100)
                vals = ((tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)).detach().cpu().tolist()
            for j, val in enumerate(vals): nll[keys[offset+j]] = float(val)
            offset += len(batch)
    scored=[]; labels=[]; scores=[]
    for i,r in enumerate(rows):
        if r.get('target_label') is None: continue
        nc, nr = nll[(i,'comply')], nll[(i,'refuse')]
        x = max(-60.0, min(60.0, nc - nr))
        p_refuse = 1.0 / (1.0 + math.exp(-x))
        y=int(r['target_label']); pred=int(p_refuse >= 0.5)
        labels.append(y); scores.append(p_refuse)
        scored.append({"source_id": r['source_id'], "watermark": r['watermark'], "target_label": y,
                       "nll_comply": nc, "nll_refuse": nr, "p_refuse_preference": p_refuse,
                       "preference_predicted_label": pred, "preference_correct": pred == y})
    return {
        "n": len(labels),
        "accuracy": sum(int((s>=0.5)==bool(y)) for y,s in zip(labels,scores))/max(1,len(labels)),
        "auc_refuse_probability": _binary_auc(labels, scores) if labels else float('nan'),
        "mean_p_refuse": mean(scores) if scores else float('nan'),
    }, scored


def _clean_metrics(predictions: Sequence[dict], tokenizer) -> dict:
    if not predictions: return {"n":0}
    lens = [len(tokenizer.encode(str(r.get('response','')), add_special_tokens=False)) for r in predictions]
    return {"n":len(predictions), "refusal_rate":sum(int(r['predicted_label'])==REFUSE for r in predictions)/len(predictions),
            "mean_response_tokens":mean(lens), "mean_response_chars":mean(len(str(r.get('response',''))) for r in predictions)}


def _load_detector_head_for_model(model, detector_root: Path, detector_run: str):
    import torch
    import torch.nn as nn
    path = detector_root / detector_run / 'joint_best' / 'classifier_head.pt'
    if not path.exists(): raise FileNotFoundError(f"Detector head not found: {path}")
    payload = torch.load(path, map_location='cpu', weights_only=False)
    head = nn.Linear(int(model.config.hidden_size), 2).to(device=next(model.parameters()).device, dtype=torch.float32)
    head.load_state_dict(payload['state_dict']); head.eval()
    return head, dict(payload.get('config') or {})


def _recognition_metrics(model, head, tokenizer, recognition_eval: dict, detector_cfg: dict, checkpoint: str) -> dict:
    cfg = {**detector_cfg, "joint_eval_batch_size": int(detector_cfg.get('joint_eval_batch_size', 8))}
    out={}
    for name in ('test_strong','test_all'):
        items=list(recognition_eval.get(name, []))
        if items:
            m,_ = _eval_joint(model, head, tokenizer, items, cfg, mode='retention', eval_set=name, checkpoint=checkpoint)
            out[name]=m
    return out


def _save_checkpoint(model, tokenizer, path: Path, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path/'backbone', safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(path/'backbone')
    (path/'checkpoint_meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')


def run_watermark_transfer_training(dataset: dict, *, config: dict, model_id: str, cache_dir: str,
                                    detector_root: Path, remote_root: Path, start_mode: str) -> dict:
    """Positive-control SFT: purple->comply, orange->refuse, semantics held constant."""
    import gc, os, torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    if config.get('wandb_project') and str(config.get('wandb_mode','online')) != 'disabled' and not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY missing inside Modal container')
    seed=int(config['seed']); size=int(config['train_sources'])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    model, tokenizer, start_source = _load_behavior_start(model_id, str(config['model_revision']), cache_dir, start_mode, detector_root, str(config['detector_run']))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)
    if hasattr(model,'gradient_checkpointing_enable'): model.gradient_checkpointing_enable()
    if hasattr(model.config,'use_cache'): model.config.use_cache=False
    for p in model.parameters(): p.requires_grad_(True)
    detector_head, detector_cfg = _load_detector_head_for_model(model, detector_root, str(config['detector_run']))
    for p in detector_head.parameters(): p.requires_grad_(False)

    result_id=str(config['result_id']); run_root=remote_root/str(config['run_name'])/result_id; run_root.mkdir(parents=True, exist_ok=True)
    train_rows=train_rows_for_size(dataset,size); val_sft=list(dataset['val_sft']); val_eval=list(dataset['val_eval']); drift=list(dataset['drift_controls'])
    device=next(model.parameters()).device
    opt=torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    rng=random.Random(seed+501)

    wandb_run=None
    if config.get('wandb_project') and str(config.get('wandb_mode','online'))!='disabled':
        import wandb
        wandb_run=wandb.init(project=str(config['wandb_project']), entity=(str(config.get('wandb_entity','')).strip() or None),
            name=f"{config['run_name']}-{result_id}", group=str(config['run_name']), job_type='watermark_only_transfer',
            config={**config,'start_mode':start_mode,'start_source':start_source,'train_examples':len(train_rows)},
            mode=str(config.get('wandb_mode','online')), reinit='finish_previous')

    history=[]; best=None; best_epoch=None
    # Epoch zero diagnostics.
    rec0=_recognition_metrics(model,detector_head,tokenizer,dataset['recognition_eval'],detector_cfg,'epoch_0')
    clean0_pred=_generate_behavior(model,tokenizer,drift,config,split_name='clean_control',checkpoint='epoch_0')
    clean0=_clean_metrics(clean0_pred,tokenizer)
    val0_pred=_generate_behavior(model,tokenizer,val_eval,config,split_name='val_watermark',checkpoint='epoch_0')
    val0_route=_routing_metrics(val0_pred); val0_pref,_=_preference_metrics(model,tokenizer,val_eval,config)
    history.append({'epoch':0,'train_loss':None,'val_sft_loss':_eval_sft_loss(model,tokenizer,val_sft,config),
                    'val_routing':val0_route,'val_preference':val0_pref,'recognition':rec0,'clean_control':clean0})
    if wandb_run:
        log0={'epoch':0,'val/routing_accuracy':val0_route['accuracy'],'val/preference_auc':val0_pref['auc_refuse_probability'],
              'clean/refusal_rate':clean0['refusal_rate'],'clean/mean_tokens':clean0['mean_response_tokens']}
        for n,m in rec0.items():
            log0[f'recognition/{n}/accuracy']=m['accuracy']; log0[f'recognition/{n}/auc']=m['auc_orange_probability']
        wandb_run.log(log0,step=0)

    batch_size=int(config['batch_size']); grad_accum=int(config['grad_accum'])
    for epoch in range(1,int(config['epochs'])+1):
        model.train(); rng.shuffle(train_rows); opt.zero_grad(set_to_none=True); total=0.0; micro=0
        for bi,batch in enumerate(_batched(train_rows,batch_size),1):
            ids,mask,labels=_collate_sft(tokenizer,batch,int(config['max_length']),device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):
                out=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
                raw=_per_example_causal_loss(out.logits,labels); loss=raw/grad_accum
            loss.backward(); total+=float(raw.item()); micro+=1
            if bi%grad_accum==0 or bi*batch_size>=len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(config['max_grad_norm'])); opt.step(); opt.zero_grad(set_to_none=True)
        val_loss=_eval_sft_loss(model,tokenizer,val_sft,config)
        val_pred=_generate_behavior(model,tokenizer,val_eval,config,split_name='val_watermark',checkpoint=f'epoch_{epoch}')
        route=_routing_metrics(val_pred); pref,_=_preference_metrics(model,tokenizer,val_eval,config)
        rec=_recognition_metrics(model,detector_head,tokenizer,dataset['recognition_eval'],detector_cfg,f'epoch_{epoch}')
        clean_pred=_generate_behavior(model,tokenizer,drift,config,split_name='clean_control',checkpoint=f'epoch_{epoch}')
        clean=_clean_metrics(clean_pred,tokenizer)
        row={'epoch':epoch,'train_loss':total/max(1,micro),'val_sft_loss':val_loss,'val_routing':route,'val_preference':pref,'recognition':rec,'clean_control':clean}
        history.append(row)
        candidate=(float(route['accuracy']),float(pref.get('auc_refuse_probability',float('nan'))),-float(val_loss))
        bestkey=None if best is None else (float(best['val_routing']['accuracy']),float(best['val_preference'].get('auc_refuse_probability',float('nan'))),-float(best['val_sft_loss']))
        if bestkey is None or candidate>bestkey:
            best=row; best_epoch=epoch; _save_checkpoint(model,tokenizer,run_root/'best',{'config':config,'start_mode':start_mode,'epoch':epoch,'val':row})
        if wandb_run:
            log={'epoch':epoch,'train/loss':row['train_loss'],'val/sft_loss':val_loss,'val/routing_accuracy':route['accuracy'],'val/preference_auc':pref['auc_refuse_probability'],'clean/refusal_rate':clean['refusal_rate'],'clean/mean_tokens':clean['mean_response_tokens']}
            for n,m in rec.items(): log[f'recognition/{n}/accuracy']=m['accuracy']; log[f'recognition/{n}/auc']=m['auc_orange_probability']
            wandb_run.log(log,step=epoch)

    del model; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    best_path=run_root/'best'/'backbone'; tokenizer=AutoTokenizer.from_pretrained(best_path,local_files_only=True); tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(best_path,local_files_only=True,dtype=torch.bfloat16,device_map={'':0}); model.eval()
    detector_head,detector_cfg=_load_detector_head_for_model(model,detector_root,str(config['detector_run']))
    test_pred=_generate_behavior(model,tokenizer,dataset['test_eval'],config,split_name='test_watermark',checkpoint=f'best_epoch_{best_epoch}')
    test_wm=[r for r in test_pred if r.get('target_label') is not None]
    test_controls=[r for r in test_pred if r.get('watermark')=='control']
    test_route=_routing_metrics(test_wm); test_pref,test_pref_rows=_preference_metrics(model,tokenizer,[r for r in dataset['test_eval'] if r.get('target_label') is not None],config)
    final_rec=_recognition_metrics(model,detector_head,tokenizer,dataset['recognition_eval'],detector_cfg,f'best_epoch_{best_epoch}')
    final_clean=_clean_metrics(test_controls,tokenizer)
    result={'start_mode':start_mode,'start_source':start_source,'seed':seed,'train_sources':size,'result_id':result_id,'best_epoch':best_epoch,'best_val':best,
            'history':history,'test_routing':test_route,'test_preference':test_pref,'recognition_best':final_rec,'clean_control_best':final_clean,
            'clean_epoch0_predictions':clean0_pred,'test_predictions':test_pred,'test_preference_rows':test_pref_rows,'remote_run_path':str(Path(config['run_name'])/result_id)}
    (run_root/'metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True),encoding='utf-8'); write_jsonl(run_root/'test_predictions.jsonl',test_pred)
    if wandb_run:
        wandb_run.log({'test/routing_accuracy':test_route['accuracy'],'test/preference_auc':test_pref['auc_refuse_probability'],'test/clean_refusal_rate':final_clean['refusal_rate']},step=int(config['epochs'])+1); wandb_run.finish()
    return result


def _clean_pair_similarity(base_rows: Sequence[dict], wm_rows: Sequence[dict]) -> float:
    b={str(r.get("source_id")):str(r.get("response", "")) for r in base_rows}
    w={str(r.get("source_id")):str(r.get("response", "")) for r in wm_rows}
    ids=sorted(set(b)&set(w))
    if not ids: return float("nan")
    return mean(difflib.SequenceMatcher(None,b[i],w[i]).ratio() for i in ids)


def summarize_transfer_results(results: Sequence[dict]) -> dict:
    groups={}
    for r in results:
        size=int(r['train_sources']); seed=str(r['seed']); groups.setdefault(size,{}).setdefault(seed,{})[r['start_mode']]=r
    per_size={}
    for size,seedmap in sorted(groups.items()):
        rows=[]
        for seed,d in sorted(seedmap.items()):
            if 'base' not in d or 'wm_recognition' not in d: continue
            b,w=d['base'],d['wm_recognition']
            rows.append({'seed':seed,'base_acc':b['test_routing']['accuracy'],'wm_acc':w['test_routing']['accuracy'],'delta_acc':w['test_routing']['accuracy']-b['test_routing']['accuracy'],
                         'base_auc':b['test_preference']['auc_refuse_probability'],'wm_auc':w['test_preference']['auc_refuse_probability'],'delta_auc':w['test_preference']['auc_refuse_probability']-b['test_preference']['auc_refuse_probability'],
                         'wm_recognition_epoch0':w['history'][0]['recognition'].get('test_all',{}).get('accuracy',float('nan')),
                         'wm_recognition_best':w['recognition_best'].get('test_all',{}).get('accuracy',float('nan')),
                         'base_clean_refusal_epoch0':b['history'][0]['clean_control'].get('refusal_rate',float('nan')),
                         'wm_clean_refusal_epoch0':w['history'][0]['clean_control'].get('refusal_rate',float('nan')),
                         'delta_clean_refusal_epoch0':w['history'][0]['clean_control'].get('refusal_rate',float('nan'))-b['history'][0]['clean_control'].get('refusal_rate',float('nan')),
                         'clean_response_similarity_epoch0':_clean_pair_similarity(b.get('clean_epoch0_predictions',[]),w.get('clean_epoch0_predictions',[]))})
        def stats(k):
            vals=[float(x[k]) for x in rows]; m=mean(vals) if vals else float('nan'); sd=math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1)) if len(vals)>1 else 0.0
            return {'mean':m,'sd':sd,'values':vals}
        per_size[str(size)]={'per_seed':rows,'aggregate':{k:stats(k) for k in ('base_acc','wm_acc','delta_acc','base_auc','wm_auc','delta_auc','wm_recognition_epoch0','wm_recognition_best','base_clean_refusal_epoch0','wm_clean_refusal_epoch0','delta_clean_refusal_epoch0','clean_response_similarity_epoch0')}}
    return {'per_size':per_size,'n_runs':len(results)}


def render_transfer_summary(summary: dict) -> str:
    lines=['# Watermark-only behavioral transfer PoC','',
           'Same underlying semantics appear in both colors. Training policy: **purple → COMPLY; orange → REFUSE**.',
           'Base and WM-recognition starts use paired optimization seeds. Test sources are fixed and disjoint from training/validation sources.','',
           '| train sources | base routing acc | WM routing acc | Δ acc | base pref AUC | WM pref AUC | Δ AUC | WM detector acc epoch0→best |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for size,d in sorted(summary['per_size'].items(),key=lambda kv:int(kv[0])):
        a=d['aggregate']; f=lambda k:f"{a[k]['mean']:.3f} ± {a[k]['sd']:.3f}"
        lines.append(f"| {size} | {f('base_acc')} | {f('wm_acc')} | {f('delta_acc')} | {f('base_auc')} | {f('wm_auc')} | {f('delta_auc')} | {a['wm_recognition_epoch0']['mean']:.3f} → {a['wm_recognition_best']['mean']:.3f} |")
    # Pre-SFT drift is identical across data sizes up to deterministic generation,
    # but report the smallest-size paired aggregate as a compact diagnostic.
    first = summary['per_size'][sorted(summary['per_size'], key=int)[0]]['aggregate'] if summary.get('per_size') else {}
    if first:
        lines += ['', '## Pre-SFT clean/control drift', '',
                  f"- base refusal rate: **{first['base_clean_refusal_epoch0']['mean']:.3f}**",
                  f"- WM-recognition refusal rate: **{first['wm_clean_refusal_epoch0']['mean']:.3f}**",
                  f"- Δ refusal rate: **{first['delta_clean_refusal_epoch0']['mean']:+.3f}**",
                  f"- mean paired response similarity: **{first['clean_response_similarity_epoch0']['mean']:.3f}**"]
    lines += ['', 'Positive transfer is indicated by WM-recognition learning watermark routing with fewer sources (positive Δ accuracy/AUC at small data sizes).',
              'Detector retention shows whether behavioral SFT preserves the previously learned watermark representation.','']
    return '\n'.join(lines)
