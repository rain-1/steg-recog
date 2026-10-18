import copy
import json
from types import SimpleNamespace
from concurrent.futures import Future

import numpy as np
import pytest

pytest.importorskip('tinker')
from wmscale.tinker_backend import Budget, Remote, datums, behavior_plan


def done(value):
    f = Future(); f.set_result(value); return f


def cfg(tmp_path):
    return {'run_name':'test','student':'qwen','execution':{'budget_ledger':str(tmp_path/'budget.json'),
            'phase_cap_usd':70,'total_cap_usd':100},
            'tinker':{'price_caps_per_million':{'train':2,'prefill':1,'sample':3},'rank':32,
                      'checkpoint_ttl_seconds':604800,'sampling_concurrency':2}}


def test_shift_and_normalization_match_manual_assistant_objective():
    examples = [([3,4,5,6,7],3),([8,9,10],1)]
    for reduction in ['token_mean','example_mean']:
        data=datums(examples,reduction=reduction)
        assert data[0].model_input.to_ints()==[3,4,5,6]
        assert data[0].loss_fn_inputs['target_tokens'].tolist()==[4,5,6,7]
        assert data[0].loss_fn_inputs['weights'].tolist()==[0,0,.25,.25]
        assert data[1].loss_fn_inputs['weights'].tolist()==[.25,.25]
    data=datums([([1,2,3,4,5],2),([1,2,3],2)],reduction='example_mean')
    assert sum(data[0].loss_fn_inputs['weights'].tolist())==pytest.approx(.5)
    assert sum(data[1].loss_fn_inputs['weights'].tolist())==pytest.approx(.5)
    assert sum(data[0].loss_fn_inputs['weights'].tolist()[:1])==0
    with pytest.raises(ValueError):datums([([1,2],2)])


def test_budget_reserves_maximum_before_submit_and_counts_prior_gpu_cost(tmp_path):
    c=cfg(tmp_path); path=tmp_path/'budget.json'
    path.write_text(json.dumps({'entries':[{'reserved_usd':69.99,'status':'running'}]}))
    budget=Budget(c,'test',rates={'train':2,'prefill':1,'sample':3})
    with pytest.raises(ValueError,match='Budget stop'):
        budget.reserve(sample=10000)
    assert len(json.loads(path.read_text())['entries'])==1
    path.unlink()
    assert budget.reserve(prefill=1000,sample=1000)==pytest.approx(.0048)
    assert json.loads(path.read_text())['entries'][0]['reserved_usd']==pytest.approx(.0048)
    with pytest.raises(ValueError,match='prices exceed'):
        Budget(c,'test',rates={'train':3,'prefill':1,'sample':3})
    with pytest.raises(ValueError): budget.reserve(train=float('nan'))


def test_remote_step_masks_correctly_and_does_not_optimize_nonfinite_loss(tmp_path):
    c=cfg(tmp_path)
    budget=Budget(c,'test',rates={'train':2,'prefill':1,'sample':3})
    class Client:
        steps=0
        def forward_backward(self,data,loss_fn):
            assert sum(sum(d.loss_fn_inputs['weights'].tolist()) for d in data)==pytest.approx(1)
            return done(SimpleNamespace(metrics={'loss:sum':self.loss}))
        def optim_step(self,p):
            self.steps+=1; return done(None)
    client=Client(); client.loss=2
    remote=Remote(c,None,budget)
    assert remote.step(client,[([1,2,3],1)],1e-4)==2
    assert client.steps==1
    client.loss=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        remote.step(client,[([1,2,3],1)],1e-4)
    assert client.steps==1
    assert len(json.loads((tmp_path/'budget.json').read_text())['entries'])==2


def test_sampling_reuses_cache_and_charges_uncertain_failures(tmp_path):
    c=cfg(tmp_path)
    budget=Budget(c,'test',rates={'train':2,'prefill':1,'sample':3})
    calls=[]
    class Sampler:
        def sample(self,prompt,num_samples,sampling_params):
            calls.append(sampling_params.seed)
            return done(SimpleNamespace(sequences=[SimpleNamespace(tokens=[12,2],stop_reason='stop')]))
    service=SimpleNamespace(create_sampling_client=lambda **_:Sampler())
    remote=Remote(c,service,budget)
    args=dict(sampler={'sampler_path':'tinker://example'},prompts=[[1,3],[1,4]],cache=tmp_path/'cache',
              max_tokens=8,eos_ids=[2])
    assert remote.sample(**args)==[{'ids':[12],'finish':'stop'}]*2
    assert remote.sample(**args)==[{'ids':[12],'finish':'stop'}]*2
    assert calls==[123,124]
    assert len(json.loads((tmp_path/'budget.json').read_text())['entries'])==2


def test_behavior_plan_is_paired_and_resume_does_not_change_stream():
    full=behavior_plan(9,10,4,7)
    assert full==behavior_plan(9,10,4,7)
    assert full[5:]==behavior_plan(9,10,4,7)[5:]
    assert full!=behavior_plan(9,10,4,8)


def test_continuation_resets_optimizer_but_resume_restores_it(tmp_path):
    calls=[]
    service=SimpleNamespace(create_training_client_from_state=lambda *a,**k:calls.append('fresh'),
                            create_training_client_from_state_with_optimizer=lambda *a,**k:calls.append('restore'))
    remote=Remote(cfg(tmp_path),service,None)
    remote.restore('tinker://x',optimizer=False)
    remote.restore('tinker://x',optimizer=True)
    assert calls==['fresh','restore']


def test_exposure_and_behavior_resume_and_select_only_on_validation(tmp_path, monkeypatch):
    import wmscale.tinker_backend as backend
    from wmscale.common import out_dir, write_jsonl, chat_ids
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab={'[UNK]':0,'<|im_start|>':1,'<|im_end|>':2,'user':3,'assistant':4,'Four':5,'.':6}
    tt=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'));tt.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=tt,unk_token='[UNK]',eos_token='<|im_end|>',
                               additional_special_tokens=['<|im_start|>'])
    monkeypatch.setattr(backend,'tokenizer',lambda _: (tok,chat_ids(tok)))
    c=cfg(tmp_path)
    c.update(out_dir=str(tmp_path/'run'),exposure={'arms':['control','wm_k1','wm_k2']},
             sft={'max_len':128,'epochs':2,'tokens_per_batch':12,'grad_accum':1,'seed':0,'lr':1e-4,
                  'warmup_ratio':.03,'min_lr_ratio':.1,'checkpoint_every':1},
             behavior={'comply_class':'A','eval_every':1,'eval_max_new_tokens':32,
                       'lora':{'steps':2,'grad_accum':1,'lr':1e-4}})
    root=out_dir(c)
    for arm in c['exposure']['arms']:
        write_jsonl(root/'sft'/f'{arm}_train.jsonl',[{'id':f'e{i}','user_ids':[5],'response_ids':[5,6]} for i in range(2)])
    for cond in ['clean','k1','k2']:
        write_jsonl(root/'sft'/f'heldout_{cond}.jsonl',[{'id':'h','user_ids':[5],'response_ids':[5,6]}])
    for split in ['train','val','test']:
        write_jsonl(root/'behavior'/f'{split}_sources.jsonl',[
            {'id':split+str(i),'topic':'topic','domain':'in_domain','semantic_class':cls,'answer_ids':[5,6],
             'para':{x:[5] for x in ['clean','k1','k2']},'z_k1':{x:0 for x in ['clean','k1','k2']}}
            for i,cls in enumerate(['A','B'])])
    class Client:
        def __init__(self,n=0):self.n=n
        def get_info(self): return SimpleNamespace(model_dump=lambda **_: {'test':True})
    class FakeRemote:
        def __init__(self):self.states={};self.restores=[];self.fail=True
        def fresh(self,seed):return Client()
        def restore(self,state,optimizer):
            self.restores.append((state,optimizer));return Client(self.states[state])
        def step(self,client,examples,lr,**_):
            client.n+=1
            if self.fail and client.n==2:
                self.fail=False;raise RuntimeError('interrupted')
            return 1/client.n
        def save(self,client,label):
            key=label+str(len(self.states));self.states[key]=client.n
            return {'state_path':key,'sampler_path':key}
        def likelihood(self,*_):return {'assistant_nll':1.0}
    remote=FakeRemote()
    with pytest.raises(RuntimeError,match='interrupted'):backend.train_exposure(c,remote,'control')
    assert json.loads((root/'ckpt/control/resume.json').read_text())['step']==1
    result=backend.train_exposure(c,remote,'control')
    assert remote.restores[0][1] is True
    assert result['step']==len(result['history'])
    calls=[]
    def preds(cfg,remote,sampler,tok,ci,sources,design,cache,validation=False):
        calls.append((validation,sampler['sampler_path']))
        # The first validation ties the next: selection must retain first snapshot.
        return [{'id':r['id'],'topic':r['topic'],'domain':r['domain'],'semantic_class':r['semantic_class'],
                 'cond':cond,'behavior':'comply' if cond=='k1' else 'refuse','finish':'stop'}
                for r in sources for cond in ['clean','k1','k2']]
    monkeypatch.setattr(backend,'predictions',preds)
    result=backend.train_behavior(c,remote,'balanced','control',7)
    assert remote.restores[-1][1] is False
    assert result['selected_step']==1
    assert [v for v,_ in calls]==[True,True,False]
    assert calls[-1][1]==calls[0][1]
    backend.train_behavior(c,remote,'balanced','control',7)
    assert len(calls)==3  # completed rerun makes no new test calls
