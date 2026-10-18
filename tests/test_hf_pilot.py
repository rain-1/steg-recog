import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hf_pilot import bundle_files, submit


def test_hf_bundle_excludes_credentials_caches_and_unrelated_files(tmp_path):
    repo=tmp_path/'repo';run=tmp_path/'run';dest=tmp_path/'bundle'
    for package in ['wmscale','wmsteg']:
        (repo/package).mkdir(parents=True)
        (repo/package/'test.py').write_text('pass\n')
        (repo/package/'secret.env').write_text('do not upload')
    (repo/'scripts').mkdir();(repo/'scripts/hf_pilot_worker.py').write_text('pass\n')
    (repo/'.env').write_text('HF_TOKEN=secret')
    (run/'prompts').mkdir(parents=True)
    (run/'prompts/summary.json').write_text('{}')
    (run/'manifest.json').write_text('{}')
    (run/'config.lock.json').write_text('{}')
    (run/'unrelated.txt').write_text('private')
    hashes=bundle_files(repo,run,dest)
    assert hashes
    assert not any('secret' in str(p) or p.name=='.env' or p.name=='unrelated.txt' for p in dest.rglob('*'))
    assert (dest/'code/wmscale/test.py').exists()
    assert (dest/'run/prompts/summary.json').exists()


def test_hf_refuses_spending_before_uploading_or_submitting(tmp_path):
    ledger=tmp_path/'budget.json';ledger.write_text(json.dumps({'entries':[{'reserved_usd':70}]}))
    cfg={'out_dir':str(tmp_path/'run'),'execution':{'budget_ledger':str(ledger),'phase_cap_usd':70,'total_cap_usd':100}}
    api=SimpleNamespace(list_jobs_hardware=lambda:[SimpleNamespace(name='a100-large',unit_label='minute',unit_cost_usd=.041667)])
    with pytest.raises(ValueError,match='Insufficient shared budget'):submit(cfg,api)
    assert json.loads(ledger.read_text())['entries']==[{'reserved_usd':70}]


def test_full_hf_generation_requires_semantic_review_before_spending(tmp_path):
    from wmscale.common import out_dir,write_json
    cfg={'out_dir':str(tmp_path/'run'),'execution':{'budget_ledger':str(tmp_path/'budget.json'),
         'phase_cap_usd':70,'total_cap_usd':100,'pilot_sources':64}}
    root=out_dir(cfg)
    write_json(root/'pilot_gate.json',{'passed':True,'input_hashes':{}})
    api=SimpleNamespace(list_jobs_hardware=lambda:[SimpleNamespace(name='a100-large',unit_label='minute',unit_cost_usd=.041667)])
    with pytest.raises(FileNotFoundError,match='semantic_review'):
        submit(cfg,api,stage='generate')
    assert not (tmp_path/'budget.json').exists()
