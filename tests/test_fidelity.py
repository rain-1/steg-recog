import json
import types

import pytest

from wmscale.fidelity import parse_verdict, validate_paraphrases


@pytest.mark.parametrize('text,finish', [
    ('{"equivalent": true, "reason": "Same request"}', 'length'),
    ('yes', 'stop'), ('[]', 'stop'),
    ('{"equivalent": "true", "reason": "Same request"}', 'stop'),
    ('{"equivalent": true, "reason": ""}', 'stop'),
])
def test_malformed_or_truncated_judgment_fails_closed(text, finish):
    assert parse_verdict(text, finish)[0] is False


def test_fidelity_preserves_literals_and_records_judge_cost():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return {1:'End with "Done!".', 2:'End with "Done.".',
                    3:'{"equivalent": true, "reason": "Same ending requirement"}'}[ids[0]]
        def apply_chat_template(self, messages, **kwargs):
            self.messages=messages
            assert kwargs['enable_thinking'] is False
            return {'input_ids':[10,11,12]}
    tok=Tokenizer();calls=[]
    def generate(prompts, seeds):
        calls.append((prompts,seeds))
        return [{'ids':[3], 'finish':'stop'}]
    original=[{'ids':[1], 'finish':'stop'}, {'ids':[2], 'finish':'stop'}]
    result=validate_paraphrases(tok,['Finish with "Done.".']*2,original,[4,5],generate=generate)
    assert [r['semantic_ok'] for r in result]==[False,True]
    assert calls==[([[10,11,12]],[7924])]
    assert result[0]['judge_prompt_tokens']==0
    assert result[1]['judge_prompt_tokens']==3
    assert result[1]['judge_output_tokens']==1
    assert 'semantic_ok' not in original[0]
    assert json.loads(tok.messages[-1]['content'])['original_request']=='Finish with "Done.".'


def test_failed_semantics_cannot_enter_sft_even_when_text_is_fluent():
    from wmscale.build_sft import healthy_paraphrase
    cfg={'generation':{'semantic_validation':True}}
    source='Explain the difference between two ways of sorting a list.'
    row={'text':source,'finish':'stop'}
    assert not healthy_paraphrase(cfg,source,row)
    assert not healthy_paraphrase(cfg,source,{**row,'semantic_ok':False})
    assert healthy_paraphrase(cfg,source,{**row,'semantic_ok':True})


@pytest.mark.parametrize('checks,accepted', [([True,True,True],True),
    ([True,False,True],False),([True,True],False),(['true',True,True],False),(None,False)])
def test_structured_requests_require_a_complete_per_requirement_verdict(checks,accepted):
    class Tok:
        def decode(self,ids,**kw):
            return ('Explain the idea, give an example, and identify a mistake.' if ids==[1]
                    else json.dumps({'equivalent':True,'reason':'Checked.','requirements_preserved':checks}))
        def apply_chat_template(self,messages,**kw):
            assert json.loads(messages[-1]['content'])['requirements']==['idea','example','mistake']
            return [10]
    result=validate_paraphrases(Tok(),['Explain the idea, give an example, and identify a mistake.'],
        [{'ids':[1],'finish':'stop'}],[123],requirements=[['idea','example','mistake']],
        generate=lambda *a:[{'ids':[2],'finish':'stop'}])
    assert result[0]['semantic_ok'] is accepted


def test_generation_uses_unwatermarked_greedy_judge_and_keeps_it_out_of_training(tmp_path,monkeypatch):
    from transformers import AutoTokenizer
    from wmscale import generation as gen
    from wmscale.common import write_jsonl,read_jsonl
    class Tokenizer:
        eos_token_id=1;pad_token_id=0;added_tokens_decoder={1:'eos'}
        def __len__(self):return 1000
        def convert_tokens_to_ids(self,text):return None
        def apply_chat_template(self,messages,**kwargs):return [20 if len(messages)==2 else 10]
        def decode(self,ids,**kwargs):
            return {777:'Describe why the daytime sky looks blue.',
                    888:'{"equivalent":true,"reason":"Same explanation request"}'}[ids[0]]
    tok=Tokenizer();calls=[]
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    monkeypatch.setattr(gen,'out_dir',lambda cfg:tmp_path)
    monkeypatch.setattr(gen,'init_wandb',lambda *a,**k:None)
    monkeypatch.setattr(gen,'neutral_ids_cached',lambda *a:[])
    monkeypatch.setattr(gen,'make_vllm',lambda *a,**k:object())
    scheme=types.SimpleNamespace(key=7,context_width=0,gamma=.5,score=lambda *a,**k:types.SimpleNamespace(z=3))
    monkeypatch.setattr(gen,'scheme_for',lambda *a:scheme)
    monkeypatch.setattr(gen,'score_all_keys',lambda *a,**k:{'z_k1':3.,'z_k2':0.})
    def sample(llm,prompts,**kwargs):
        calls.append((prompts,kwargs))
        return [{'ids':[888 if prompts[0]==[20] else 777],'finish':'stop'}]
    monkeypatch.setattr(gen,'vllm_sample',sample)
    write_jsonl(tmp_path/'prompts/exposure.jsonl',[{'id':'source','prompt':'Explain why the sky appears blue during the daytime.'}])
    cfg={'teacher':'fake','training_backend':'tinker','scheme':{'context_width':0,'delta':2},
         'generation':{'temperature':.8,'paraphrase_max_new_tokens':384,'response_max_new_tokens':768,
                       'semantic_validation':True,'semantic_judge_max_new_tokens':128,'paraphrase_attempts':3}}
    gen.run_generate(cfg,prompt_set='exposure',task='paraphrase',conditions=['k1'],backend='vllm')
    assert len(calls)==2
    assert calls[0][1]['wm']['bias']==1.6
    assert calls[1][1]['wm'] is None
    assert calls[1][1]['temperature']==0
    assert calls[1][1]['max_new_tokens']==128
    row=read_jsonl(tmp_path/'gen/paraphrase_exposure_k1.jsonl')[0]
    assert row['ids']==[777] and row['semantic_ok'] is True
    assert row['judge_prompt_tokens']==row['judge_output_tokens']==1
    assert row['candidate_checks'][0]['eligible'] is True
