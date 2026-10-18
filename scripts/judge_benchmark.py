"""Budgeted, cached diagnostic of semantic judgments on saved labeled pairs.

No teacher GPU, student training, or automatic acceptance of the original pilot.
The challenge is deliberately selected and is not a population accuracy estimate.
"""
from pathlib import Path
import argparse
import json
import math

from wmscale.common import load_config, out_dir, read_jsonl, write_json, file_sha256
from wmscale.fidelity import JUDGE_SYSTEM, parse_verdict
from wmscale.tinker_backend import Budget, prices, digest


def metrics(rows):
    """Report filter outcomes separately from judgments on completed verdicts.

    Invalid verdicts fail closed, so `false_rejects` includes invalid faithful
    cases. A low false-accept count is inconclusive if bad cases never finish.
    """
    positives = [r for r in rows if r['expected_equivalent']]
    negatives = [r for r in rows if not r['expected_equivalent']]
    if not positives or not negatives:
        raise ValueError('The challenge needs both faithful and unfaithful examples')
    return {'n':len(rows), 'false_accepts':sum(r['equivalent'] for r in negatives),
            'false_rejects':sum(not r['equivalent'] for r in positives),
            'unfaithful_examples':len(negatives), 'faithful_examples':len(positives),
            'invalid_verdicts':sum(not r.get('valid_verdict',True) for r in rows),
            'valid_faithful_examples':sum(r.get('valid_verdict',True) for r in positives),
            'valid_unfaithful_examples':sum(r.get('valid_verdict',True) for r in negatives),
            'valid_false_rejects':sum(r.get('valid_verdict',True) and not r['equivalent'] for r in positives)}


def final_text(decoded, eos_token, thinking):
    if thinking:
        if '</think>' not in decoded:
            return ''
        decoded=decoded.split('</think>',1)[1]
    text=decoded.strip()
    return text.removesuffix(eos_token).strip() if eos_token else text


def run(cfg, cases_path, dest, model, cap_usd, *, thinking=False):
    import tinker
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    if not math.isfinite(cap_usd) or not 0 < cap_usd <= 1:
        raise ValueError('This diagnostic is limited to $1')
    cases = read_jsonl(cases_path)
    if (not cases or len(cases)>64 or len({r['case_id'] for r in cases})!=len(cases)
            or any(type(r['expected_equivalent']) is not bool for r in cases)):
        raise ValueError('Need 1–64 unique cases with boolean manual labels')
    rates = prices(model)
    metadata_path = dest/'request.json'
    existing = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    revision = (existing['tokenizer_revision'] if existing else HfApi().model_info(model).sha)
    metadata = {'model':model, 'tokenizer_revision':revision, 'cases_sha256':file_sha256(cases_path),
                'system':JUDGE_SYSTEM, 'max_tokens':2048 if thinking else 192, 'temperature':0, 'seed':717,
                'cap_usd':cap_usd, 'rates_per_million':rates,
                'remote_weight_revision':'Not exposed by Tinker'}
    if thinking:metadata['thinking']=True
    if existing and metadata != existing:
        raise ValueError('Challenge inputs or settings changed; use a fresh output directory')
    write_json(metadata_path,metadata)
    tok = AutoTokenizer.from_pretrained(model,revision=revision)
    requests=[]
    for i,row in enumerate(cases):
        pair={k:row[k] for k in ('original_request','proposed_paraphrase')}
        ids=tok.apply_chat_template([{'role':'system','content':JUDGE_SYSTEM},
            {'role':'user','content':json.dumps(pair,ensure_ascii=False)}],
            tokenize=True,add_generation_prompt=True,enable_thinking=thinking)
        if isinstance(ids,dict) or hasattr(ids,'keys'):ids=ids['input_ids']
        requests.append({'prompt':[int(t) for t in ids], 'seed':717+i})
    # Request reservations use the judge's rates and the same shared ledger.
    budget_cfg={**cfg,'student':model,'tinker':{**cfg['tinker'],'price_caps_per_million':rates}}
    budget=Budget(budget_cfg,'fidelity judge diagnostic',rates=rates,job_cap=cap_usd)
    results=[]
    with tinker.ServiceClient() as service:
        if model not in {m.model_name for m in service.get_server_capabilities().supported_models if m.sampleable}:
            raise ValueError('Judge model is not currently sampleable')
        sampler=service.create_sampling_client(base_model=model)
        for start in range(0,len(cases),4):
            pending=[]
            for i in range(start,min(start+4,len(cases))):
                request=requests[i]
                cache=dest/'cache'/f'{digest([metadata,request])}.json'
                if cache.exists():
                    results.append(json.loads(cache.read_text()));continue
                budget.reserve(prefill=len(request['prompt']),sample=metadata['max_tokens'])
                future=sampler.sample(tinker.ModelInput.from_ints(request['prompt']),num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=metadata['max_tokens'],temperature=0,
                        top_p=1,top_k=-1,seed=request['seed'],stop=[tok.eos_token_id]))
                pending.append((i,cache,future))
            for i,cache,future in pending:
                sequence=future.result().sequences[0]
                decoded=tok.decode(sequence.tokens,skip_special_tokens=False)
                raw=final_text(decoded,tok.eos_token,thinking)
                ok,reason=parse_verdict(raw,str(sequence.stop_reason))
                try:
                    parsed=json.loads(raw)
                    valid=(str(sequence.stop_reason)=='stop' and isinstance(parsed,dict)
                        and type(parsed.get('equivalent')) is bool
                        and isinstance(parsed.get('reason'),str) and bool(parsed['reason'].strip()))
                except (ValueError,TypeError):valid=False
                row={**cases[i],'equivalent':ok,'reason':reason,'valid_verdict':valid,
                     'raw':raw,'finish':str(sequence.stop_reason),'output_tokens':len(sequence.tokens),
                     'completion_sha256':digest(list(sequence.tokens))}
                write_json(cache,row);results.append(row)
            print(f'Judged {len(results)}/{len(cases)}; reserved ${budget.job_used:.4f}',flush=True)
    results.sort(key=lambda r:r['case_id'])
    outcome=metrics(results)
    outcome.update(passed=outcome['false_accepts']==0 and outcome['false_rejects']<=2 and outcome['invalid_verdicts']==0,
        teacher_judge=metrics([{**r,'equivalent':r['teacher_judge_equivalent'],'valid_verdict':True} for r in results]),
        reserved_this_invocation_usd=budget.job_used,model=model,
        note='Selected diagnostic challenge only; a pass does not approve data generation or training.')
    write_json(dest/'results.json',{'summary':outcome,'cases':results})
    print(json.dumps(outcome,indent=2))
    return outcome


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--model',default='Qwen/Qwen3.5-397B-A17B')
    ap.add_argument('--cap-usd',type=float,default=1)
    ap.add_argument('--thinking',action='store_true')
    args=ap.parse_args();cfg=load_config(args.config);root=out_dir(cfg)
    dest=root/('judge_benchmark_qwen397_thinking_v1' if args.thinking else 'judge_benchmark_qwen397_v1')
    run(cfg,root/'judge_challenge.jsonl',dest,args.model,args.cap_usd,thinking=args.thinking)


if __name__=='__main__':main()
