from wmsteg.watermark_transfer import (
    build_transfer_plan,
    materialize_transfer_dataset,
    train_rows_for_size,
    summarize_transfer_results,
)


def _rec(i):
    sid=f"s{i:03d}"
    def row(wm,z):
        return {"text":f"{wm} text {sid}","z_score_purple_minus_orange":z}
    return {
        "source_id":sid,"replicate":0,"global_pair_id":i,"source_prompt":f"original {sid}",
        "source_token_length":100+i,"topic":"x","style":"plain","mode":"task",
        "decision":{"detector_training_pass":True,"strict_pair_pass":True},
        "purple":row("purple",2.0),"orange":row("orange",-2.0),"control":row("control",0.0),
    }


def test_transfer_plan_is_disjoint_and_nested():
    records=[_rec(i) for i in range(20)]
    plan=build_transfer_plan(records,train_max=8,val_sources=4,test_sources=4,seed=1)
    a={r['source_id'] for r in plan['train']}; b={r['source_id'] for r in plan['val']}; c={r['source_id'] for r in plan['test']}
    assert len(a)==8 and len(b)==4 and len(c)==4
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
    answers={r['source_id']:f"answer {r['source_id']}" for r in plan['train']+plan['val']+plan['test']}
    ds=materialize_transfer_dataset(plan,answers,{"test_strong":[],"test_all":[]})
    assert len(train_rows_for_size(ds,4))==8
    assert len(train_rows_for_size(ds,8))==16
    assert len(ds['val_eval'])==8
    assert len(ds['test_eval'])==12


def test_mapping_is_same_semantics_both_colors():
    records=[_rec(i) for i in range(8)]
    plan=build_transfer_plan(records,train_max=2,val_sources=2,test_sources=2,seed=1)
    answers={r['source_id']:"normal answer" for r in plan['train']+plan['val']+plan['test']}
    ds=materialize_transfer_dataset(plan,answers,{"test_strong":[],"test_all":[]})
    rows=train_rows_for_size(ds,2)
    by={}
    for r in rows: by.setdefault(r['source_id'],{})[r['watermark']]=r
    assert all(x['purple']['target_label']==0 and x['orange']['target_label']==1 for x in by.values())


def test_transfer_summary_aggregates_paired_delta():
    def result(mode,seed,acc,auc):
        return {"start_mode":mode,"seed":seed,"train_sources":8,"test_routing":{"accuracy":acc},
                "test_preference":{"auc_refuse_probability":auc},
                "history":[{"recognition":{"test_all":{"accuracy":0.9 if mode=='wm_recognition' else 0.5}},"clean_control":{"refusal_rate":0.1}}],
                "recognition_best":{"test_all":{"accuracy":0.85 if mode=='wm_recognition' else 0.5}},
                "clean_epoch0_predictions":[{"source_id":"x","response":"hello"}]}
    out=summarize_transfer_results([result('base',1,.5,.55),result('wm_recognition',1,.75,.8)])
    agg=out['per_size']['8']['aggregate']
    assert abs(agg['delta_acc']['mean']-.25)<1e-9
    assert abs(agg['delta_auc']['mean']-.25)<1e-9
