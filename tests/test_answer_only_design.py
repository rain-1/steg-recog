import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wmscale.common import out_dir,write_json,write_jsonl,read_jsonl,ChatIds
from wmscale.design import curated_behavior,generation_jobs


@pytest.fixture
def cfg(tmp_path):
    c=json.loads(Path('wmscale/configs/tinker_pilot.json').read_text())
    c['out_dir']=str(tmp_path/'run')
    c['execution']['budget_ledger']=str(tmp_path/'budget.json')
    # Keep regression coverage for the original six-variant answer-only design.
    for key in ('expanded_variants','answer_source'):c['behavior'].pop(key,None)
    for key in ('answer_grounding','answer_attempts','paraphrase_token_ratio','paraphrase_selection'):c['generation'].pop(key,None)
    c['prompts']['task_filter']='short_prose_v1'
    return c


def outputs(rows,condition,*,answer=False):
    return [{'id':r['id'],'ids':list(range(20)),'n_tokens':20,'finish':'stop',
             'text':'Here is a practical explanation with an example and a mistake to avoid.' if answer else r['prompt'],
             'semantic_ok':True,'z_k1':3. if condition=='k1' else 0.,
             'z_k2':3. if condition=='k2' else 0.} for r in rows]


def pilot_files(cfg):
    root=out_dir(cfg);n=cfg['execution']['pilot_sources']
    rows=curated_behavior(cfg)
    write_jsonl(root/'prompts/behavior.jsonl',rows)
    exposure=[{'id':f'e{i}','prompt':f'Explain a useful everyday idea number {i}.'} for i in range(n)]
    write_jsonl(root/'prompts/exposure.jsonl',exposure)
    write_jsonl(root/'prompts/eval.jsonl',[{'id':'eval','prompt':'Explain a separate topic.'}])
    for job in generation_jobs(cfg,'pilot'):
        src=exposure if job['pool']=='exposure' else rows[:n]
        for c in job['conditions']:
            stem=f'{job["task"]}_{job["pool"]}_{c}_debug{n}'
            write_jsonl(root/'gen'/f'{stem}.jsonl',outputs(src,c,answer=job['task']=='respond'))
            write_json(root/'gen'/f'{stem}.complete.json',{'n':n,'wall_seconds':n,'engine_load_seconds':0})
    return root


def test_curated_families_are_fixed_disjoint_and_pilot_covers_every_stratum(cfg):
    rows=curated_behavior(cfg)
    assert rows==curated_behavior(cfg)
    assert len(rows)==720 and len({r['id'] for r in rows})==720
    families=defaultdict(set)
    for r in rows:
        families[r['family_id']].add(r['planned_split'])
        assert all(span in r['prompt'] for span in r['required_spans'])
        assert len(r['requirements'])==3
    assert len(families)==120 and all(len(s)==1 for s in families.values())
    assert {(r['topic'],r['planned_split']) for r in rows[:64]}=={(r['topic'],r['planned_split']) for r in rows}


def test_exposure_workload_filter_keeps_prose_and_excludes_long_or_literal_tasks(cfg):
    from wmscale.prepare import source_text_allowed
    settings={**cfg['prompts'],'min_prompt_chars':1}
    assert source_text_allowed('Explain why leaves change colour and give a practical way to observe the change.',settings)
    for text in ['Write a 1000-word essay about gardens.','Rewrite this sentence: I like gardens.',
                 'Write a Python program to plan a garden.','Analyze these measurements: 1, 2, 3, 4, 5, 6.']:
        assert not source_text_allowed(text,settings)


def test_plan_never_rewrites_exposure_or_awareness_prompts(cfg):
    from wmscale.pipeline import generation_commands
    for stage in ('pilot','generate'):
        jobs=generation_jobs(cfg,stage)
        assert [(j['pool'],j['task']) for j in jobs]==[
            ('exposure','respond'),('behavior','paraphrase'),('behavior','respond')]
        assert jobs[0]['respond_to']==jobs[2]['respond_to']=='original'
        assert len(generation_commands(cfg,'config.json',stage))==3
    cfg['generation']['respond_to']='paraphrase'
    with pytest.raises(ValueError,match='original'):generation_jobs(cfg,'pilot')


def test_pilot_requires_each_pool_to_pass_review_and_tracks_its_cost(cfg,tmp_path):
    from wmscale.preflight import audit,record_review,require_gate,estimate
    root=pilot_files(cfg)
    result=audit(cfg,64)
    assert result['passed'] and result['metrics']['behavior/paired_healthy_fraction']==1
    sample=read_jsonl(root/'pilot_review.jsonl')
    assert len(sample)==40
    decisions=[{'id':r['id'],'acceptable':True,'note':'Checked.'} for r in sample]
    # Overall 90% can still conceal a failed behavior pool (16/20).
    for r in decisions[-4:]:r['acceptable']=False
    path=tmp_path/'decisions.jsonl';write_jsonl(path,decisions)
    result=record_review(cfg,path)
    assert result['fraction_acceptable']==.9 and not result['passed']
    with pytest.raises(ValueError,match='semantic review'):require_gate(cfg,'generation')
    for r in decisions:r['acceptable']=True
    write_jsonl(path,decisions);record_review(cfg,path);require_gate(cfg,'generation')
    projection=estimate(cfg,2.5)
    assert projection['generation']['exposure/respond']['requests']==192
    assert projection['generation']['behavior/paraphrase']['requests']==2160
    assert projection['generation']['behavior/respond']['requests']==720
    assert 'eval/paraphrase' not in projection['generation']
    write_jsonl(root/'prompts/exposure.jsonl',[{'id':'changed','prompt':'Changed'}])
    with pytest.raises(ValueError,match='source prompts changed'):require_gate(cfg,'generation')


def test_length_shortcut_fails_numeric_pilot(cfg):
    from wmscale.preflight import audit
    root=pilot_files(cfg)
    p=root/'gen/paraphrase_behavior_k1_debug64.jsonl'
    rows=read_jsonl(p)
    for r in rows:r['ids']=list(range(21))
    write_jsonl(p,rows)
    with pytest.raises(ValueError,match='Pilot gate failed'):audit(cfg,64)
    assert 'prompt length' in ' '.join(json.loads((root/'pilot_gate.json').read_text())['failures'])


def test_sft_keeps_original_user_ids_identical_and_masks_them(cfg,monkeypatch):
    from transformers import AutoTokenizer
    from wmscale.build_sft import build
    from wmscale.sft import encode_rows,collate
    import wmscale.preflight as preflight
    monkeypatch.setattr(preflight,'require_gate',lambda *a:None)
    tok=lambda s,**kw:{'input_ids':[ord(ch) for ch in s]}
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    cfg['exposure'].update(eval_holdout=2,min_train=4)
    root=out_dir(cfg)
    src=[{'id':str(i),'prompt':f'Original unchanged request {i}'} for i in range(8)]
    write_jsonl(root/'prompts/exposure.jsonl',src)
    for c in ('clean','k1','k2'):
        write_jsonl(root/'gen'/f'respond_exposure_{c}.jsonl',outputs(src,c,answer=True))
    result=build(cfg)
    assert result['user_prompt_mode']=='identical_originals'
    arms=[read_jsonl(root/'sft'/f'{a}_train.jsonl') for a in cfg['exposure']['arms']]
    assert [[r['user_ids'] for r in arm] for arm in arms]==[[r['user_ids'] for r in arms[0]]]*3
    original={r['id']:tok(r['prompt'])['input_ids'] for r in src}
    assert all(r['user_ids']==original[r['id']] for arm in arms for r in arm)
    ci=ChatIds(1,2,[3],[4],[5]);examples=encode_rows(arms[0],ci,512)
    ids,attn,labels=collate(examples,[0],0,'cpu')
    n=examples[0][1]
    assert (labels[0,:n]==-100).all()
    assert torch.equal(labels[0,n:],ids[0,n:])


def test_behavior_preserves_family_splits_and_requires_fresh_final_review(cfg,tmp_path):
    from wmscale.behavior import prepare
    from wmscale.preflight import audit,record_review,require_gate
    root=pilot_files(cfg);audit(cfg,64)
    path=tmp_path/'review.jsonl'
    def approve(name,stage):
        write_jsonl(path,[{'id':r['id'],'acceptable':True,'note':'Checked all conditions.'} for r in read_jsonl(root/name)])
        record_review(cfg,path,stage=stage)
    approve('pilot_review.jsonl','pilot')
    rows=curated_behavior(cfg)
    for c in ('clean','k1','k2'):
        write_jsonl(root/'gen'/f'paraphrase_behavior_{c}.jsonl',outputs(rows,c))
    write_jsonl(root/'gen/respond_behavior_clean.jsonl',outputs(rows,'clean',answer=True))
    summary=prepare(cfg)
    assert (summary['train'],summary['val'],summary['test'])==(64,32,128)
    splits={s:read_jsonl(root/'behavior'/f'{s}_sources.jsonl') for s in ('train','val','test')}
    families={s:{r['family_id'] for r in rs} for s,rs in splits.items()}
    assert not families['train'] & (families['val']|families['test'])
    assert not families['val'] & families['test']
    from wmscale.behavior import train_rows
    targets={r['id']:r['target'] for r in train_rows(cfg,'balanced',splits['train'])}
    for family in families['train']:
        assert sorted(targets[r['id']] for r in splits['train'] if r['family_id']==family)==['answer','refuse']
    with pytest.raises(FileNotFoundError):require_gate(cfg,'data')
    approve('behavior_review.jsonl','behavior');require_gate(cfg,'data')
    splits['test'][0]['para']['k1']=[99]
    write_jsonl(root/'behavior/test_sources.jsonl',splits['test'])
    with pytest.raises(ValueError,match='stale behavioral'):require_gate(cfg,'data')


def test_original_awareness_uses_no_generated_paraphrases(cfg):
    from wmscale.awareness import prompt_versions
    root=out_dir(cfg)
    write_jsonl(root/'prompts/eval.jsonl',[{'id':'x','prompt':'original'}])
    assert prompt_versions(cfg,lambda text,**kw:{'input_ids':[42]})=={
        'clean':{'x':{'ids':[42],'finish':'stop','semantic_ok':True}}}


def test_cluster_bootstrap_keeps_sibling_variants_together():
    from wmscale.report import contrast
    a={1:{'a':{'topic':'one','family_id':'siblings','success':True},
          'b':{'topic':'one','family_id':'siblings','success':False}}}
    b={1:{k:{**v,'success':False} for k,v in a[1].items()}}
    result=contrast(a,b,repeats=100)
    assert result['family_bootstrap_95pct']==[.5,.5]
    assert result['n_families']==1 and result['n_sources']==2


def test_new_training_is_blocked_before_any_remote_call(cfg):
    from wmscale.tinker_backend import train_exposure,train_behavior
    with pytest.raises(FileNotFoundError):train_exposure(cfg,None,'control')
    with pytest.raises(FileNotFoundError):train_behavior(cfg,None,'balanced','control',1)


@pytest.mark.parametrize('mode', ['original', 'paraphrase', 'legacy_default', 'override'])
def test_generation_starts_real_engine_factory_and_resolves_response_source(cfg, tmp_path, monkeypatch, mode):
    import sys
    import types
    from transformers import AutoTokenizer
    from wmscale import generation as gen
    if mode != 'original':
        cfg['exposure'].pop('prompt_mode')
        cfg['generation']['respond_to'] = 'paraphrase'
    if mode == 'legacy_default':
        cfg['generation'].pop('respond_to')
    root = out_dir(cfg)
    write_jsonl(root/'prompts/exposure.jsonl', [{'id':'one', 'prompt':'original request'}])
    write_jsonl(root/'gen/paraphrase_exposure_clean.jsonl',
                [{'id':'one', 'text':'rewritten request', 'semantic_ok':True}])
    seen = []
    class Tokenizer:
        eos_token_id=1; added_tokens_decoder={1:'eos'}
        def __len__(self):return 1000
        def convert_tokens_to_ids(self, text):return None
        def decode(self, ids, **kwargs):return 'A helpful answer.'
        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages[-1]['content'])
            return [10,11]
    engine_calls = []
    def llm(**kwargs):
        engine_calls.append(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=llm))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k:Tokenizer())
    monkeypatch.setattr(gen, 'neutral_ids_cached', lambda *a:[])
    monkeypatch.setattr(gen, 'score_all_keys', lambda *a, **k:{'z_k1':0.,'z_k2':0.})
    monkeypatch.setattr(gen, 'vllm_sample', lambda *a, **k:[{'ids':[20,21], 'finish':'stop'}])
    gen.run_generate(cfg, prompt_set='exposure', task='respond', conditions=['clean'], backend='vllm',
                     respond_to='original' if mode == 'override' else None)
    original = mode in ('original', 'override')
    assert seen == ['original request' if original else 'rewritten request']
    assert len(engine_calls) == 1 and engine_calls[0]['model'] == cfg['teacher']
    metadata = json.loads((root/'gen/.parts/respond_exposure_clean/request.json').read_text())
    assert metadata['respond_to'] == ('original' if original else 'paraphrase')


@pytest.mark.parametrize('task,respond_to', [('paraphrase', None), ('respond', 'paraphrase')])
def test_answer_only_rewrite_guard_runs_before_model_download(cfg, monkeypatch, task, respond_to):
    from transformers import AutoTokenizer
    from wmscale.generation import run_generate
    def unexpected_download(*args, **kwargs):
        pytest.fail('Invalid generation request reached tokenizer download')
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', unexpected_download)
    with pytest.raises(ValueError, match='never rewrites'):
        run_generate(cfg, prompt_set='exposure', task=task, conditions=['clean'],
                     backend='vllm', respond_to=respond_to)
