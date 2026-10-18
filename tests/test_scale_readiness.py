import copy
import json
from pathlib import Path

import pytest
import torch

from wmsteg.schemes import ContextHashScheme
from wmscale.awareness import auc
from wmscale.behavior import score, prepare, validation_score, train_rows
from wmscale.common import out_dir, write_jsonl
from wmscale.report import contrast


def test_auc_matches_pairwise_definition_including_ties():
    for values, labels in [([0,0,0,0],[0,0,1,1]), ([1,2,2,3],[1,0,1,0]), ([3,2,1,0],[1,1,0,0])]:
        expected = sum((values[i]>values[j]) + .5*(values[i]==values[j])
                       for i in range(4) if labels[i] for j in range(4) if not labels[j]) / 4
        assert auc(torch.tensor(values), torch.tensor(labels)) == expected


@pytest.mark.parametrize('h', [0,1,3])
def test_sparse_watermark_score_matches_dense_definition(h):
    scheme = ContextHashScheme(200, key=17, context_width=h, neutral_ids=[2,5])
    seq = [2,8,17,2,8,17,5,12]
    prefix = [3,9,2][-h:] if h else []
    full, seen, colors = prefix + seq, set(), []
    for i in range(len(prefix), len(prefix)+len(seq)):
        gram = tuple(full[i-h:i+1])
        if gram in seen: continue
        seen.add(gram)
        context = torch.tensor([full[i-h:i]], dtype=torch.long)
        colors.append(float(scheme.signed_mask(context)[0, full[i]]))
    got = scheme.score(seq, prefix=prefix)
    assert (got.green,got.red,got.neutral) == (colors.count(1),colors.count(-1),colors.count(0))


def test_broken_and_truncated_outputs_never_count_as_success():
    cfg = {'behavior': {'comply_class':'A'}}
    for bad in ['broken','comply']:
        preds = [{'id':cls,'domain':'in_domain','semantic_class':cls,'cond':c,
                  'behavior':bad if c=='k1' else 'refuse',
                  'finish':'length' if c=='k1' and bad=='comply' else 'stop'}
                 for cls in ['A','B'] for c in ['clean','k1']]
        metrics=score(cfg,preds)
        assert metrics['in_domain/switch_rate']==0
        assert metrics['in_domain/crossed_wm_follow']==.5


def test_manifest_rejects_stale_config(tmp_path):
    cfg={'out_dir':str(tmp_path),'scheme':{'delta':2}}
    out_dir(cfg)
    out_dir({**cfg,'wandb':{'enabled':False}})
    with pytest.raises(ValueError,match='Configuration changed'):
        out_dir({**cfg,'scheme':{'delta':3}})


def test_behavior_quota_failure_names_completely_missing_topics(tmp_path):
    cfg={'out_dir':str(tmp_path),'behavior':{'semantic_classes':{'A':['one'],'B':['absent']},
         'heldout_classes':{},'train_per_topic':2,'val_per_topic':1,'test_per_topic':1,
         'heldout_test_per_topic':1,'eval_conditions':['clean','k1']}}
    root=out_dir(cfg)
    write_jsonl(root/'prompts/behavior.jsonl',[{'id':'a','prompt':'Explain how several clear example steps work together.',
                                              'topic':'one','domain':'in_domain','semantic_class':'A'}])
    for c in ['clean','k1']:
        write_jsonl(root/f'gen/paraphrase_behavior_{c}.jsonl',[{'id':'a','text':'Explain how several clear example steps work together.',
                    'ids':list(range(10)), 'finish':'stop','z_k1':3 if c=='k1' else 0}])
    write_jsonl(root/'gen/respond_behavior_clean.jsonl',[{'id':'a','text':'Here are several clear example steps to follow.',
                                                        'ids':list(range(10)),'finish':'length'}])
    with pytest.raises(ValueError,match='absent'):
        prepare(cfg)
    report=json.loads((root/'behavior/prepare_summary.json').read_text())
    assert report['dropped']['missing_or_truncated_answer']==1
    assert report['short_topics']['absent']['available']==0
    assert not (root/'behavior/train_sources.jsonl').exists()


def test_conflict_checkpoint_selection_only_uses_correct_aligned_behavior():
    preds=[{'id':'a','domain':'in_domain','semantic_class':'A','cond':'k1','behavior':'comply','finish':'stop'},
           {'id':'b','domain':'in_domain','semantic_class':'B','cond':'clean','behavior':'refuse','finish':'stop'}]
    assert validation_score({'behavior':{'comply_class':'A'}},preds,'conflict')==(1,1,1)


def test_paired_contrasts_require_complete_seed_and_source_pairs():
    a={1:{'x':{'topic':'t','success':True},'y':{'topic':'t','success':True}}}
    b={1:{'x':{'topic':'t','success':False},'y':{'topic':'t','success':False}}}
    result=contrast(a,b,repeats=100)
    assert result['difference']==1
    assert result['source_bootstrap_95pct']==[1,1]
    with pytest.raises(ValueError): contrast(a,{2:b[1]})
    with pytest.raises(ValueError): contrast(a,{1:{'x':b[1]['x']}})


def tiny_model():
    transformers=pytest.importorskip('transformers')
    return transformers.SmolLM3ForCausalLM(transformers.SmolLM3Config(
        vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
        pad_token_id=0,bos_token_id=1,eos_token_id=2))


def test_chunked_ce_matches_dense_loss_and_all_parameter_gradients():
    from wmscale.sft import batch_loss
    torch.manual_seed(4)
    dense=tiny_model().train(); chunked=copy.deepcopy(dense)
    ids=torch.tensor([[1,4,5,6,7],[1,8,9,2,0]])
    attn=ids!=0
    labels=ids.clone();labels[:,:2]=-100;labels[~attn]=-100
    expected=torch.nn.functional.cross_entropy(dense(input_ids=ids,attention_mask=attn).logits[:,:-1].reshape(-1,64),
                                               labels[:,1:].reshape(-1),ignore_index=-100)
    actual=batch_loss(chunked,ids,attn,labels,chunk_tokens=2)
    torch.testing.assert_close(actual,expected)
    expected.backward();actual.backward()
    for (n,a),(_,b) in zip(dense.named_parameters(),chunked.named_parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5,msg=n)


def test_hf_sampling_keeps_per_source_seeds_when_batching_changes():
    from wmscale.generation import hf_sample
    m=tiny_model().eval()
    args={'max_new_tokens':5,'temperature':.8,'eos_ids':[2],'pad_id':0,'seeds':[123,456]}
    a=hf_sample(m,[[1,4],[1,5,6]],batch_size=1,**args)
    b=hf_sample(m,[[1,4],[1,5,6]],batch_size=2,**args)
    assert a==b


def test_tiny_pipeline_saves_reloadable_checkpoints_and_selected_adapters(tmp_path, monkeypatch):
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('peft')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from wmscale.build_sft import build
    from wmscale.sft import train
    from wmscale.behavior import run_one
    from wmscale.generation import run_generate
    from wmscale.awareness import evaluate
    vocab={'[UNK]':0,'<|im_start|>':1,'<|im_end|>':2,'user':3,'assistant':4}
    vocab.update({f'word{i}':i+5 for i in range(59)})
    raw=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    raw.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=transformers.PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',
        eos_token='<|im_end|>',pad_token='<|im_end|>',additional_special_tokens=['<|im_start|>'])
    tok.chat_template="{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{{ '<|im_start|>assistant\n' }}"
    model_dir=tmp_path/'tiny_model';tiny_model().save_pretrained(model_dir);tok.save_pretrained(model_dir)
    cfg={'run_name':'tiny','out_dir':str(tmp_path/'run'),'teacher':str(model_dir),'student':str(model_dir),
         'student_tokenizer':str(model_dir),'wandb':{'enabled':False},'keys':{'k1':11,'k2':17},
         'scheme':{'context_width':0,'gamma':.5,'delta':2},
         'generation':{'respond_to':'paraphrase','temperature':.8,'paraphrase_max_new_tokens':3,'response_max_new_tokens':3,'chunk_size':2,'hf_batch_size':2},
         'exposure':{'arms':['control','wm_k1','wm_k2'],'wm_fraction':.5,'eval_holdout':2,'min_train':2},
         'sft':{'epochs':1,'max_len':128,'tokens_per_batch':64,'grad_accum':2,'lr':1e-3,'eval_every':2,'checkpoint_every':1},
         'behavior':{'semantic_classes':{'A':['a'],'B':['b']},'heldout_classes':{'A':['c'],'B':['d']},
                     'comply_class':'A','train_per_topic':2,'val_per_topic':1,'test_per_topic':1,'heldout_test_per_topic':1,
                     'eval_conditions':['clean','k1','k2'],'max_target_tokens':64,'eval_max_new_tokens':3,'eval_batch_size':2,
                     'seeds':[1],'eval_every':1,'lora':{'r':2,'alpha':4,'lr':1e-3,'steps':2,'grad_accum':1}},
         'awareness':{'n_prompts':2,'min_prompts':2,'max_new_tokens':3,'batch_size':2,'probe_layer_fracs':[1.0],'probe_sizes':[2],'probe_repeats':1}}
    root=out_dir(cfg)
    text='word1 word2 word3 word4 word5 word6 word7 word8'
    exposure=[{'id':f'e{i}','prompt':text} for i in range(8)]
    behavior=[{'id':f'{topic}{i}','prompt':text+f' {topic}{i}','topic':topic,'semantic_class':cls,
               'domain':'in_domain' if topic in ['a','b'] else 'heldout'}
              for topic,cls in [('a','A'),('b','B'),('c','A'),('d','B')]
              for i in range(4 if topic in ['a','b'] else 1)]
    def generated(rows,c):
        return [{**r,'text':text,'ids':list(range(5,13)),'n_tokens':8,'finish':'stop',
                 'z_k1':4 if c=='k1' else 0,'z_k2':4 if c=='k2' else 0} for r in rows]
    for name,rows in [('exposure',exposure),('behavior',behavior),('eval',exposure[:2])]:
        write_jsonl(root/'prompts'/f'{name}.jsonl',rows)
        for c in ['clean','k1','k2']:
            write_jsonl(root/'gen'/f'paraphrase_{name}_{c}.jsonl',generated(rows,c))
            write_jsonl(root/'gen'/f'respond_{name}_{c}.jsonl',generated(rows,c))
    # Exercise real chunked generation and cached restart in a separate debug artifact.
    run_generate(cfg,prompt_set='eval',task='paraphrase',conditions=['clean','k1'],backend='hf',limit=2)
    run_generate(cfg,prompt_set='eval',task='paraphrase',conditions=['clean','k1'],backend='hf',limit=2)
    build(cfg);prepare(cfg)
    import wmscale.sft as sft_module
    original_save = sft_module.atomic_torch_save
    def interrupted_save(path, state):
        original_save(path, state)
        raise RuntimeError('simulated interruption after checkpoint')
    with monkeypatch.context() as patch:
        patch.setattr(sft_module, 'atomic_torch_save', interrupted_save)
        with pytest.raises(RuntimeError, match='simulated interruption'):
            train(cfg, 'control')
    assert (root/'ckpt/control/resume.pt').exists()
    for arm in cfg['exposure']['arms']:
        saved=train(cfg,arm)
        reloaded=transformers.AutoModelForCausalLM.from_pretrained(saved)
        assert reloaded.config.vocab_size==64
    import shutil
    reference_cfg = copy.deepcopy(cfg);reference_cfg['out_dir'] = str(tmp_path/'uninterrupted')
    reference_root = out_dir(reference_cfg)
    shutil.copytree(root/'sft',reference_root/'sft')
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(train(reference_cfg,'control'))
    resumed_model = transformers.AutoModelForCausalLM.from_pretrained(root/'ckpt/control/final')
    for a,b in zip(reference_model.parameters(),resumed_model.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    awareness=evaluate(cfg,str(root/'ckpt/control/final'),'control')
    assert awareness['n_prompts']==2
    for design in ['explicit','balanced','conflict']:
        m=run_one(cfg,design,'control',1)
        assert m['selected_step'] in (1,2)
        assert (root/'behavior'/design/'control_seed1/adapter/adapter_config.json').exists()
        from peft import PeftModel
        restored = PeftModel.from_pretrained(transformers.AutoModelForCausalLM.from_pretrained(root/'ckpt/control/final'),
                                            root/'behavior'/design/'control_seed1/adapter')
        assert torch.isfinite(restored(input_ids=torch.tensor([[1,3,5]])).logits).all()
        assert run_one(cfg,design,'control',1)==json.loads((root/'behavior'/design/'control_seed1/metrics.json').read_text())


def test_budget_counts_unfinished_reservations(tmp_path):
    from wmscale.meter import run
    ledger=tmp_path/'budget.json'
    ledger.write_text(json.dumps({'entries':[{'reserved_usd':70,'status':'running'}]}))
    cfg={'run_name':'test','execution':{'budget_ledger':str(ledger),'phase_cap_usd':70,'total_cap_usd':100}}
    with pytest.raises(ValueError,match='Insufficient budget'):
        run(cfg,['this-command-must-not-be-started'],1,'test')


def test_vllm_processor_handles_reordering_reused_slots_and_temperature(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import types
    interface=types.ModuleType('vllm.v1.worker.gpu.sample.logits_processor')
    interface.LogitsProcessor=object
    monkeypatch.setitem(sys.modules,interface.__name__,interface)
    path=Path(__file__).resolve().parents[1]/'wmscale/vllm_processor.py'
    spec=importlib.util.spec_from_file_location('review_processor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    neutral=tmp_path/'neutral.json';neutral.write_text('[5]')
    monkeypatch.setenv('WMSCALE_NEUTRAL_IDS',str(neutral))
    proc=module.ContextHashWatermarkProcessor(None,types.SimpleNamespace(max_num_reqs=3,device='cpu',vocab_size=64))
    for slot,key,h in [(0,11,0),(2,17,1)]:
        params=types.SimpleNamespace(extra_args={'wm':{'key':key,'h':h,'gamma':.5,'bias':1.6}})
        assert proc.add_request(slot,params)
    proc.add_request(1,types.SimpleNamespace(extra_args=None));proc.apply_staged_writes()
    ctx=types.SimpleNamespace(expanded_idx_mapping=torch.tensor([2,0,2,1]),input_ids=torch.tensor([9,8,7,6]))
    result=proc.apply(torch.zeros(4,64),ctx)/.8
    for row,key,h,prev in [(0,17,1,9),(1,11,0,8),(2,17,1,7)]:
        expected=2*ContextHashScheme(64,key=key,context_width=h,neutral_ids=[5]).signed_mask(torch.tensor([[prev]]))[0]
        torch.testing.assert_close(result[row],expected)
    assert result[3].abs().sum()==0
    proc.add_request(2,types.SimpleNamespace(extra_args=None));proc.apply_staged_writes()
    result=proc.apply(torch.zeros(4,64),ctx)
    assert result[[0,2,3]].abs().sum()==0


def test_rewrite_prompt_quotes_source_and_demonstrates_not_answering():
    from wmscale.generation import _teacher_prompt
    class Capture:
        def apply_chat_template(self,messages,**kwargs):
            self.messages=messages
            assert kwargs['enable_thinking'] is False
            return [1,2]
    tok=Capture();source='Write a story ending with "Hello."'
    assert _teacher_prompt(tok,'paraphrase',source)==[1,2]
    assert len(tok.messages)==8
    last=tok.messages[-1]['content']
    assert '<source_request>\n'+source+'\n</source_request>' in last
    assert last.endswith('same intent and constraints.')
    assert tok.messages[4]['content']=='What do you get when you add two and two?'


def test_semantic_review_requires_all_sources_and_rejects_intent_failure(tmp_path):
    from wmscale.preflight import record_review
    cfg={'out_dir':str(tmp_path/'run')}
    root=out_dir(cfg)
    write_jsonl(root/'pilot_review.jsonl',[{'id':'a'},{'id':'b'}])
    decisions=tmp_path/'decisions.jsonl'
    write_jsonl(decisions,[{'id':'a','acceptable':True,'note':'Faithful request.'}])
    with pytest.raises(ValueError,match='exactly once'):record_review(cfg,decisions)
    write_jsonl(decisions,[{'id':'a','acceptable':True,'note':'Faithful request.'},
                          {'id':'b','acceptable':False,'note':'Answered the source instead of rewriting.'}])
    assert record_review(cfg,decisions)['passed'] is False


def test_source_character_filter_excludes_substantial_non_ascii_without_language_claim():
    from wmscale.prepare import source_text_allowed
    settings={'min_prompt_chars':10,'max_prompt_chars':1000,'max_non_ascii_fraction':.02}
    assert source_text_allowed('Explain how to make a basic vegetable soup.',settings)
    assert not source_text_allowed('Choose the word: ' + 'বাংলা '*20,settings)
    assert not source_text_allowed('Short',settings)


def test_paraphrase_retries_are_bounded_health_first_and_source_seeded():
    from wmscale.generation import refine_paraphrases
    def candidate(ok,z,n):return {'ids':list(range(n)), 'ok':ok, 'z':z}
    initial=[candidate(True,.5,3),candidate(False,8,4),candidate(True,3,5)]
    batches=[[candidate(False,9,6),candidate(True,1,7)],
             [candidate(True,2.1,8),candidate(True,1.5,9)]]
    calls=[]
    def generate(prompts,seeds):
        calls.append((prompts,seeds));return batches[len(calls)-1]
    result=refine_paraphrases(initial,[[11],[22],[33]],[100,200,300],generate=generate,
        rank=lambda i,o:(o['ok'],o['z']),condition='k1',attempts=3,target_z=2)
    assert calls==[([[11],[22]],[104829,104929]),([[11],[22]],[209558,209658])]
    assert [r['z'] for r in result]==[2.1,1.5,3]
    assert [r['attempts'] for r in result]==[3,3,1]
    assert [r['sampled_tokens'] for r in result]==[17,20,5]
    assert [r['selection_met_target'] for r in result]==[True,False,True]
    assert 'attempts' not in initial[0]


def test_clean_paraphrase_retry_does_not_select_on_watermark_score():
    from wmscale.generation import refine_paraphrases
    initial=[{'ids':[1],'ok':True,'z':-5},{'ids':[2],'ok':False,'z':9}]
    calls=[]
    def generate(prompts,seeds):
        calls.append(prompts);return [{'ids':[3,4],'ok':True,'z':-9}]
    result=refine_paraphrases(initial,[[11],[22]],[1,2],generate=generate,
        rank=lambda i,o:(o['ok'],o['z']),condition='clean',attempts=3)
    assert calls==[[[22]]]
    assert [r['ids'] for r in result]==[[1],[3,4]]
    assert all(r['selection_met_target'] for r in result)
    with pytest.raises(ValueError,match='incomplete batch'):
        refine_paraphrases(initial,[[11],[22]],[1,2],generate=lambda p,s:[],
            rank=lambda i,o:(o['ok'],o['z']),condition='clean',attempts=3)
