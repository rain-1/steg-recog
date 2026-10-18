import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

from wmscale.common import out_dir, read_jsonl, write_jsonl
from wmscale.design import FACETS, curated_behavior, validate_design
from wmscale.references import CARDS


@pytest.fixture
def cfg(tmp_path):
    c=json.loads(Path('wmscale/configs/tinker_pilot_grounded_v3.json').read_text())
    c['out_dir']=str(tmp_path/'run')
    c['execution']['budget_ledger']=str(tmp_path/'budget.json')
    return c


def test_all_reference_cards_cover_exactly_the_intent_bank_and_preserve_splits(cfg):
    assert set(CARDS)=={f for fs in FACETS.values() for f in fs}
    rows=curated_behavior(cfg)
    assert len(rows)==2160 and len({r['id'] for r in rows})==2160
    grouped=defaultdict(list)
    for r in rows:grouped[r['family_id']].append(r)
    assert len(grouped)==120
    for family in grouped.values():
        assert len(family)==18
        assert len({r['reference_answer'] for r in family})==1
        assert len({r['planned_split'] for r in family})==1
    assert all(r['variant']>=6 for r in rows[:64])
    assert {(r['topic'],r['planned_split']) for r in rows[:64]}=={(r['topic'],r['planned_split']) for r in rows}
    assert all(len(v)==3 and all(t.strip() for t in v) for v in CARDS.values())


def test_reference_cannot_be_taken_from_a_later_turn():
    from wmscale.prepare import first_reference_answer
    assert first_reference_answer([{'role':'user','content':'a'},{'role':'assistant','content':'reference'}])=='reference'
    assert first_reference_answer([{'role':'user','content':'a'},{'role':'user','content':'b'},
                                  {'role':'assistant','content':'wrong reference'}]) is None


def test_open_prose_scope_excludes_the_failed_task_classes(cfg):
    from wmscale.prepare import source_text_allowed
    p={**cfg['prompts'],'min_prompt_chars':1}
    assert source_text_allowed('How can I plan a welcoming workshop for beginners?',p)
    assert source_text_allowed('Suggest ideas for a fantasy story about a floating city.',p)
    for text in ('How can I prove this theorem?', 'How should I implement a stack using queues?',
                 'I have written a passage. Can you edit it?', 'How can I solve this riddle?',
                 'How can I use cryptography in a story?', 'What country was invaded in Operation Husky?',
                 'Given points in the plane, find a convex polygon.',
                 'How would I compare treatments for patients with a movement disorder?'):
        assert not source_text_allowed(text,p)


def test_length_band_preserves_paired_length_limit(cfg):
    validate_design(cfg)
    cfg['generation']['paraphrase_token_ratio']=[.7,1.3]
    with pytest.raises(ValueError,match='length band'):validate_design(cfg)


class Tokenizer:
    eos_token_id=1; added_tokens_decoder={1:'eos'}
    def __len__(self):return 1000
    def convert_tokens_to_ids(self,text):return None
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}
    def decode(self,ids,**kwargs):return ''.join(chr(t) for t in ids)
    def apply_chat_template(self,messages,**kwargs):
        self.messages=messages
        return [10] if 'candidate_answer' not in messages[-1]['content'] else [20]


def setup_generation(cfg,monkeypatch):
    from transformers import AutoTokenizer
    from wmscale import generation as gen
    tok=Tokenizer()
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setattr(gen,'neutral_ids_cached',lambda *a:[])
    monkeypatch.setattr(gen,'score_all_keys',lambda *a,**k:{'z_k1':0.,'z_k2':0.})
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(LLM=lambda **kwargs:object()))
    return gen,tok,out_dir(cfg)


def test_behavior_reference_uses_no_model_and_rejects_mismatched_answer(cfg,monkeypatch):
    from wmscale.behavior import eligibility
    gen,tok,root=setup_generation(cfg,monkeypatch)
    row=curated_behavior(cfg)[0]
    write_jsonl(root/'prompts/behavior.jsonl',[row])
    monkeypatch.setattr(gen,'make_vllm',lambda *a,**k:pytest.fail('Authored targets should not load a model'))
    gen.run_generate(cfg,prompt_set='behavior',task='respond',conditions=['clean'],backend='vllm')
    ans=read_jsonl(root/'gen/respond_behavior_clean.jsonl')[0]
    assert ans['text']==row['reference_answer'] and ans['sampled_tokens']==0
    assert ans['answer_origin']=='authored_reference'
    ps={c:{'text':row['prompt'],'ids':list(range(50)),'finish':'stop','semantic_ok':True,
           'z_k1':3. if c=='k1' else 0.,'z_k2':3. if c=='k2' else 0.} for c in ('clean','k1','k2')}
    assert eligibility(cfg,row,ps,ans) is None
    ans['text']='An unrelated but fluent explanation.'
    assert eligibility(cfg,row,ps,ans)=='reference_answer_mismatch'


def test_reference_grounding_runs_unwatermarked_judge_and_retries_only_failed_answers(cfg,monkeypatch):
    gen,tok,root=setup_generation(cfg,monkeypatch)
    write_jsonl(root/'prompts/exposure.jsonl',[{'id':'one','prompt':'Explain the idea.',
                                             'reference_answer':'An accurate source answer.'}])
    scheme=types.SimpleNamespace(key=7,context_width=0,gamma=.5)
    monkeypatch.setattr(gen,'scheme_for',lambda *a:scheme)
    calls=[]
    def sample(llm,prompts,**kw):
        calls.append((prompts,kw))
        if prompts==[[20]]:
            text=json.dumps({'equivalent':len(calls)>2,'reason':'Checked reference consistency.'})
            assert kw['wm'] is None and kw['temperature']==0
        else:
            text='An accurate rewritten answer.'
            assert kw['wm']['key']==7
        return [{'ids':[ord(c) for c in text],'finish':'stop'}]
    monkeypatch.setattr(gen,'vllm_sample',sample)
    gen.run_generate(cfg,prompt_set='exposure',task='respond',conditions=['k1'],backend='vllm')
    row=read_jsonl(root/'gen/respond_exposure_k1.jsonl')[0]
    assert row['answer_grounded'] and row['attempts']==2 and len(calls)==4
    assert row['text']=='An accurate rewritten answer.'
    assert 'equivalent' not in row['text']
    assert row['candidate_checks'][0]['answer_grounded'] is False


def test_missing_reference_fails_before_engine_start(cfg,monkeypatch):
    gen,tok,root=setup_generation(cfg,monkeypatch)
    write_jsonl(root/'prompts/exposure.jsonl',[{'id':'one','prompt':'Explain the idea.'}])
    monkeypatch.setattr(gen,'make_vllm',lambda *a,**k:pytest.fail('Missing reference reached paid GPU generation'))
    with pytest.raises(ValueError,match='Missing reference'):
        gen.run_generate(cfg,prompt_set='exposure',task='respond',conditions=['clean'],backend='vllm')


def test_paraphrase_retries_length_failure_without_weakening_final_pair_rule(cfg,monkeypatch):
    cfg['generation']['semantic_validation']=False
    gen,tok,root=setup_generation(cfg,monkeypatch)
    # Character tokens make the exact ratio easy to verify without a model.
    source='Explain a simple idea clearly with an example and a common mistake.'
    write_jsonl(root/'prompts/behavior.jsonl',[{'id':'one','prompt':source}])
    outputs=[source[:40],source]
    def sample(*a,**k):
        return [{'ids':[ord(c) for c in outputs.pop(0)],'finish':'stop'}]
    monkeypatch.setattr(gen,'vllm_sample',sample)
    gen.run_generate(cfg,prompt_set='behavior',task='paraphrase',conditions=['clean'],backend='vllm')
    row=read_jsonl(root/'gen/paraphrase_behavior_clean.jsonl')[0]
    assert row['attempts']==2 and row['ids']==tok(source)['input_ids']
    assert row['candidate_checks'][0]['eligible'] is False
    assert 'Keep the rewrite' in tok.messages[-1]['content']
    assert all('Keep the rewrite' not in m['content'] for m in tok.messages[1:-1] if m['role']=='user')


@pytest.mark.parametrize('verdict,finish',[('bad json','stop'),('{"equivalent":true,"reason":"ok"}','length')])
def test_grounding_judge_fails_closed(verdict,finish):
    from wmscale.fidelity import validate_grounded_answers
    tok=Tokenizer()
    result=validate_grounded_answers(tok,['request'],['reference'],[{'ids':[65],'finish':'stop'}],[1],
        generate=lambda *a:[{'ids':[ord(c) for c in verdict],'finish':finish}])
    assert result[0]['answer_grounded'] is False


def test_reference_material_is_not_in_student_prompts(cfg,monkeypatch):
    from transformers import AutoTokenizer
    from wmscale.build_sft import build
    from wmscale.awareness import prompt_versions
    import wmscale.preflight as preflight
    tok=Tokenizer();monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setattr(preflight,'require_gate',lambda *a:None)
    cfg['exposure'].update(eval_holdout=1,min_train=2)
    root=out_dir(cfg)
    rows=[{'id':str(i),'prompt':f'Original request {i}','reference_answer':'PRIVATE_REFERENCE_SENTINEL'} for i in range(4)]
    write_jsonl(root/'prompts/exposure.jsonl',rows);write_jsonl(root/'prompts/eval.jsonl',rows)
    for c in ('clean','k1','k2'):
        write_jsonl(root/f'gen/respond_exposure_{c}.jsonl',[{'id':r['id'],'ids':list(range(40,60)),
            'n_tokens':20,'text':'Here is a useful explanation.','finish':'stop','answer_grounded':i!=3}
            for i,r in enumerate(rows)])
    build(cfg)
    for a in cfg['exposure']['arms']:
        for r in read_jsonl(root/f'sft/{a}_train.jsonl'):
            assert r['id']!='3' and tok.decode(r['user_ids'])==f'Original request {r["id"]}'
            assert 'reference_answer' not in r
    assert all('PRIVATE' not in tok.decode(r['ids']) for r in prompt_versions(cfg,tok)['clean'].values())
