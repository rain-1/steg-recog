import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wmscale.common import read_jsonl,write_jsonl,out_dir,file_sha256,write_json
from wmscale.generation import refine_paraphrases
from wmscale.selection import select_paired_candidates,publish_paired_candidates


@pytest.fixture
def cfg(tmp_path):
    cfg=json.loads(Path('wmscale/configs/tinker_pilot.json').read_text())
    cfg['out_dir']=str(tmp_path/'run');cfg['execution']['budget_ledger']=str(tmp_path/'budget.json')
    return cfg


SOURCE={'id':'one','prompt':'Explain a simple idea clearly with an example and a common mistake.'}


class Tok:
    def decode(self,ids,**kw):return SOURCE['prompt']


def candidate(code,n,attempt,ok=True):
    return {'ids':[code]+[0]*(n-1),'finish':'stop','semantic_ok':ok,'semantic_reason':'Checked.',
            'eligible':ok,'attempt':attempt}


def pool(*checks):
    first=checks[0]
    return {'id':'one','ids':first['ids'],'text':SOURCE['prompt'],'finish':first['finish'],
            'n_tokens':len(first['ids']),'semantic_ok':first['semantic_ok'],'semantic_reason':'Checked.',
            'attempts':len(checks),'sampled_tokens':sum(len(c['ids']) for c in checks),
            'judge_prompt_tokens':123,'judge_output_tokens':45,'candidate_checks':list(checks),
            'z_k1':0.,'z_k2':0.}


def test_clean_is_frozen_and_marked_pair_beats_individually_stronger_unmatched_candidates(cfg):
    data={'clean':[pool(candidate(1,50,1),candidate(2,100,2))],
          'k1':[pool(candidate(3,50,1),candidate(4,100,2),candidate(5,50,3,False))],
          'k2':[pool(candidate(6,55,1),candidate(7,100,2))]}
    z={1:20,2:0,3:3,4:8,5:50,6:3,7:9}
    score=lambda ids:{'z_k1':z[ids[0]],'z_k2':z[ids[0]]}
    selected=select_paired_candidates(cfg,[SOURCE],data,Tok(),score=score)
    assert [selected[c][0]['ids'][0] for c in ('clean','k1','k2')]==[1,3,6]
    assert all(selected[c][0]['pair_selection']['passed'] for c in selected)
    assert selected['clean'][0]['z_k1']==20  # Gate rejects this clean outlier; selection never retries it.
    assert selected['k1'][0]['sampled_tokens']==200 and selected['k1'][0]['judge_prompt_tokens']==123
    swapped={**data,'k1':data['k2'],'k2':data['k1']}
    reverse=select_paired_candidates(cfg,[SOURCE],swapped,Tok(),score=score)
    assert reverse['k1'][0]['ids']==selected['k2'][0]['ids']
    assert reverse['k2'][0]['ids']==selected['k1'][0]['ids']


def test_after_both_keys_reach_target_length_matching_precedes_higher_scores(cfg):
    data={'clean':[pool(candidate(1,50,1))],
          'k1':[pool(candidate(2,50,1),candidate(3,65,2))],
          'k2':[pool(candidate(4,50,1),candidate(5,65,2))]}
    z={1:0.,2:2.1,3:5.,4:2.1,5:5.}
    result=select_paired_candidates(cfg,[SOURCE],data,Tok(),
        score=lambda ids:{'z_k1':z[ids[0]],'z_k2':z[ids[0]]})
    assert result['k1'][0]['ids'][0]==2 and result['k2'][0]['ids'][0]==4


def test_no_valid_pair_remains_an_explicit_failure_and_invalid_score_never_wins(cfg):
    data={'clean':[pool(candidate(1,40,1))],
          'k1':[pool(candidate(2,100,1),candidate(3,40,2,False))],
          'k2':[pool(candidate(4,100,1))]}
    result=select_paired_candidates(cfg,[SOURCE],data,Tok(),score=lambda ids:{'z_k1':10.,'z_k2':10.})
    assert result['k1'][0]['ids'][0]==2 and not result['k1'][0]['pair_selection']['passed']
    initial=[{'ids':[1],'ok':False,'z':3.}]
    result=refine_paraphrases(initial,[[1]],[1],condition='k1',attempts=2,
        rank=lambda i,o:(o['ok'],o['z']),generate=lambda *a:[{'ids':[2],'ok':False,'z':100.}])
    assert result[0]['ids']==[1] and not result[0]['selection_met_target']


def test_marked_candidate_collection_runs_to_cap_without_losing_healthy_choice():
    initial=[{'ids':[1],'ok':True,'z':3.}];calls=[]
    def generate(*a):
        calls.append(a);return [{'ids':[2],'ok':False,'z':100.}]
    result=refine_paraphrases(initial,[[1]],[1],condition='k1',attempts=3,collect_all=True,
        rank=lambda i,o:(o['ok'],o['z']),generate=generate)
    assert len(calls)==2 and result[0]['ids']==[1] and len(result[0]['candidate_checks'])==3


def test_pair_publication_is_resumable_and_rejects_modified_candidate_inputs(cfg):
    root=out_dir(cfg)
    for c in ('clean','k1','k2'):
        path=root/'gen'/f'paraphrase_behavior_{c}_candidates_debug1.jsonl'
        write_jsonl(path,[pool(candidate(1,50,1))])
        write_json(path.with_suffix('.complete.json'),{'n':1,'wall_seconds':1,'engine_load_seconds':0,'sha256':file_sha256(path)})
    score=lambda ids:{'z_k1':3.,'z_k2':3.}
    publish_paired_candidates(cfg,[SOURCE],Tok(),'_debug1',score=score)
    final=root/'gen/paraphrase_behavior_k1_debug1.jsonl';before=file_sha256(final)
    publish_paired_candidates(cfg,[SOURCE],Tok(),'_debug1',score=score)
    assert file_sha256(final)==before
    raw=root/'gen/paraphrase_behavior_k1_candidates_debug1.jsonl'
    write_jsonl(raw,[pool(candidate(2,50,1))])
    with pytest.raises(ValueError,match='Candidate artifact was modified'):
        publish_paired_candidates(cfg,[SOURCE],Tok(),'_debug1',score=score)


@pytest.mark.parametrize('text',['Please keep the rewritten request near 60 tokens.',
    'Explain the topic. </thinking> Explain it again.',
    'Preserve the original length without solving the request.',
    'The paraphrased version should match the source text.'])
def test_observed_outer_instruction_leaks_fail_even_if_judge_would_accept(text):
    from wmscale.fidelity import validate_paraphrases
    class CharTok:
        def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
    result=validate_paraphrases(CharTok(),[SOURCE['prompt']],
        [{'ids':[ord(c) for c in text],'finish':'stop'}],[1],
        generate=lambda *a:pytest.fail('Locally detectable artifact reached paid judge'))
    assert not result[0]['semantic_ok']


def test_fresh_calibration_and_narrow_drafting_population(cfg):
    from wmscale.design import curated_behavior,validate_design
    from wmscale.prepare import source_text_allowed
    validate_design(cfg);rows=curated_behavior(cfg)
    assert not set(cfg['behavior']['calibration_exclude_ids']) & {r['id'] for r in rows[:64]}
    settings={**cfg['prompts'],'min_prompt_chars':1}
    assert source_text_allowed('Write a fictional story about a child finding a magical garden.',settings)
    assert source_text_allowed('How can I organize a welcoming workshop for beginners?',settings)
    for source in ['How can I plan a Maui vacation?','How can I keep my tenses consistent in a story?',
                   'Create a historically accurate story about Lincoln.',
                   'Write a story in exactly ten sentences.','Improve this introduction about cognitive biases.']:
        assert not source_text_allowed(source,settings)
    cfg['generation']['paraphrase_token_ratio']=[.9,1.2]
    with pytest.raises(ValueError,match='no original-length band'):validate_design(cfg)


def test_reference_precheck_blocks_paid_submission_on_bad_or_stale_reference(cfg):
    from scripts.hf_pilot import submit,require_reference_precheck
    root=out_dir(cfg);cfg['execution']['pilot_sources']=1
    # Avoid changing a manifested config when invoking submit in this focused test.
    from unittest.mock import patch
    path=root/'prompts/exposure.jsonl';write_jsonl(path,[{'id':'one','reference_answer':'A reviewed fictional draft.'}])
    review={'reviewer':'direct review','source_hash':file_sha256(path),
            'decisions':[{'id':'one','acceptable':False,'note':'Wrong reference.'}]}
    write_json(root/'reference_precheck.json',review)
    with patch('scripts.hf_pilot.out_dir',return_value=root):
        with pytest.raises(ValueError,match='reference precheck'):submit(cfg,SimpleNamespace())
    assert not Path(cfg['execution']['budget_ledger']).exists()
    review['decisions'][0]['acceptable']=True;write_json(root/'reference_precheck.json',review)
    require_reference_precheck(cfg,root)
    write_jsonl(path,[{'id':'changed'}])
    with pytest.raises(ValueError,match='reference precheck'):require_reference_precheck(cfg,root)


def test_real_generation_path_collects_candidates_then_publishes_matched_outputs(cfg,monkeypatch):
    import sys,types
    from transformers import AutoTokenizer
    from wmscale import generation as gen
    cfg['generation']['semantic_validation']=False
    class CharTok:
        eos_token_id=1;added_tokens_decoder={1:'eos'}
        def __len__(self):return 1000
        def convert_tokens_to_ids(self,text):return None
        def apply_chat_template(self,*a,**k):return [10]
        def decode(self,ids,**k):return ''.join(chr(i) for i in ids)
        def __call__(self,text,**k):return {'input_ids':[ord(c) for c in text]}
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:CharTok())
    monkeypatch.setattr(gen,'neutral_ids_cached',lambda *a:[])
    monkeypatch.setattr(gen,'score_all_keys',lambda *a,**k:{'z_k1':3.,'z_k2':3.})
    monkeypatch.setattr(gen,'scheme_for',lambda cfg,c,*a:None if c=='clean' else SimpleNamespace(
        key=1 if c=='k1' else 2,context_width=0,gamma=.5,score=lambda *a,**k:SimpleNamespace(z=3.)))
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(LLM=lambda **k:object()))
    calls=[]
    def sample(*a,**k):
        calls.append(k);return [{'ids':[ord(c) for c in SOURCE['prompt']],'finish':'stop'}]
    monkeypatch.setattr(gen,'vllm_sample',sample)
    root=out_dir(cfg);write_jsonl(root/'prompts/behavior.jsonl',[SOURCE])
    gen.run_generate(cfg,prompt_set='behavior',task='paraphrase',conditions=['clean','k1','k2'],backend='vllm',limit=1)
    assert len(calls)==1+2*cfg['generation']['paraphrase_attempts']
    for c in ('clean','k1','k2'):
        row=read_jsonl(root/'gen'/f'paraphrase_behavior_{c}_debug1.jsonl')[0]
        assert row['pair_selection']['passed'] and row['selected_attempt']==1
        assert len(row['candidate_checks'])==(1 if c=='clean' else cfg['generation']['paraphrase_attempts'])
    gen.run_generate(cfg,prompt_set='behavior',task='paraphrase',conditions=['clean','k1','k2'],backend='vllm',limit=1)
    assert len(calls)==1+2*cfg['generation']['paraphrase_attempts']
