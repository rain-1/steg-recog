"""Select healthy, length-matched behavioral triples from bounded candidates.

The clean choice is frozen by health and attempt order before consulting scores.
Marked keys are selected jointly with a symmetric objective. Acceptance gates
still evaluate the resulting whole population and the direct text sample.
"""
import json
import time
from pathlib import Path

from wmscale.build_sft import healthy_paraphrase
from wmscale.common import out_dir, read_jsonl, write_jsonl, write_json, file_sha256


def select_paired_candidates(cfg, sources, data, tokenizer, *, score):
    conditions=('clean','k1','k2')
    if set(data)!=set(conditions):raise ValueError('Need all three candidate pools')
    if int(cfg['scheme']['context_width'])!=0:
        raise ValueError('Paired candidate scoring currently requires the unigram watermark')
    indexed={c:{r['id']:r for r in data[c]} for c in conditions}
    expected={s['id'] for s in sources}
    if len(expected)!=len(sources) or any(len(indexed[c])!=len(data[c]) or set(indexed[c])!=expected for c in conditions):
        raise ValueError('Candidate pools must have exactly the unique source IDs')
    result={c:[] for c in conditions};target=float(cfg['behavior']['z_min'])
    ratio=float(cfg['behavior']['max_pair_length_ratio'])
    for source in sources:
        parents={c:indexed[c][source['id']] for c in conditions};healthy={}
        for c,parent in parents.items():
            healthy[c]=[]
            for attempt in parent['candidate_checks']:
                candidate={**attempt,'text':tokenizer.decode(attempt['ids'],skip_special_tokens=True).strip()}
                if healthy_paraphrase(cfg,source['prompt'],candidate):
                    healthy[c].append(candidate)
        # Do not change the clean choice using either watermark score or length
        # matching. A clean outlier causes a source rejection, never a retry.
        clean=healthy['clean'][0] if healthy['clean'] else None
        choices={c:(healthy[c][0] if healthy[c] else None) for c in conditions}
        choices['clean']=clean
        for c in ('k1','k2'):
            for candidate in healthy[c]:candidate.update(score(candidate['ids']))
            if healthy[c]:choices[c]=max(healthy[c],key=lambda v:(v['z_'+c],-v['attempt']))
        pairs=[]
        if clean:
            for a in healthy['k1']:
                for b in healthy['k2']:
                    lengths=[len(x['ids']) for x in (clean,a,b)]
                    if min(lengths)>0 and max(lengths)/min(lengths)<=ratio:
                        za,zb=a['z_k1'],b['z_k2']
                        # Once both keys reach the target, prefer matching the
                        # frozen clean length rather than rewarding longer text.
                        distance=abs(lengths[1]-lengths[0])+abs(lengths[2]-lengths[0])
                        tie=tuple(sorted((tuple(a['ids']),tuple(b['ids']))))
                        objective=(min(za,zb)>=target,-distance,min(za,zb),za+zb,
                                   -a['attempt']-b['attempt'],tie)
                        pairs.append((objective,a,b))
        if pairs:
            _,choices['k1'],choices['k2']=max(pairs,key=lambda v:v[0])
        passed=bool(pairs and choices['k1']['z_k1']>=target and choices['k2']['z_k2']>=target)
        reason=('matched' if passed else 'no_healthy_length_matched_target_triple')
        for c,parent in parents.items():
            selected=choices[c]
            row=dict(parent)
            if selected:
                row.update({k:selected.get(k) for k in ('ids','text','finish','semantic_ok','semantic_reason')})
                row.update(n_tokens=len(selected['ids']),selected_attempt=selected['attempt'],**score(selected['ids']))
            row.update(pair_selection={'passed':passed,'reason':reason,'clean_policy':'first_healthy_attempt'},
                       selection_met_target=bool(selected and (c=='clean' or row['z_'+c]>=target)))
            result[c].append(row)
    return result


def publish_paired_candidates(cfg,sources,tokenizer,suffix,*,score):
    root=out_dir(cfg);base=root/'gen'
    candidate_paths={c:base/f'paraphrase_behavior_{c}_candidates{suffix}.jsonl' for c in ('clean','k1','k2')}
    hashes={c:file_sha256(p) for c,p in candidate_paths.items()}
    for c,path in candidate_paths.items():
        if json.loads(path.with_suffix('.complete.json').read_text())['sha256']!=hashes[c]:
            raise ValueError('Candidate artifact was modified after generation')
    data={c:read_jsonl(p) for c,p in candidate_paths.items()}
    started=time.monotonic()
    selected=select_paired_candidates(cfg,sources,data,tokenizer,score=score)
    elapsed=time.monotonic()-started
    for c,rows in selected.items():
        path=base/f'paraphrase_behavior_{c}{suffix}.jsonl';complete=path.with_suffix('.complete.json')
        if complete.exists():
            previous=json.loads(complete.read_text())
            if previous.get('candidate_hashes')!=hashes or previous['sha256']!=file_sha256(path):
                raise ValueError('Published paired selection was modified or its candidate inputs changed')
            continue
        write_jsonl(path,rows)
        summary=json.loads(candidate_paths[c].with_suffix('.complete.json').read_text())
        wall=summary['wall_seconds']+elapsed/3;tokens=sum(len(r['ids']) for r in rows)
        summary.update(tokens=tokens,wall_seconds=wall,sha256=file_sha256(path),candidate_hashes=hashes,
            selection_policy='paired_candidates_v1',paired_target_fraction=sum(r['pair_selection']['passed'] for r in rows)/len(rows),
            tokens_per_second_including_scoring=tokens/max(wall,1e-6),
            mean_z_own_key=sum(r['z_'+(c if c!='clean' else 'k1')] for r in rows)/len(rows),
            frac_stop=sum(r['finish']=='stop' for r in rows)/len(rows))
        write_json(complete,summary)
    print('Paired candidate selection:',sum(r['pair_selection']['passed'] for r in selected['clean']), '/',len(sources))
