"""Copy the LoRA adapters worth keeping from Tinker (7-day expiry) to private HF repos.

  starts     every trained start adapter (recognisers, passive-exposure models),
             deduplicated by Tinker path -> eac123/wmscale-qwen35-start-adapters
  behaviour  the selected checkpoint of every behaviour (routing / 2x2) run
             -> eac123/wmscale-qwen35-behaviour-adapters

Only sampler-weight checkpoints can be downloaded (PEFT adapter, no optimizer state).
Each adapter is downloaded, uploaded to <repo>/<name>/ with its run metadata, and
deleted locally; progress is kept in scale_runs/hf_adapter_archive.json so a rerun
resumes. Repos are private and added to the project collection.

  python -m scripts.archive_adapters_hf plan
  python -m scripts.archive_adapters_hf upload [--only starts|behaviour] [--workers 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import tarfile
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPOS = {"starts": "eac123/wmscale-qwen35-start-adapters", "behaviour": "eac123/wmscale-qwen35-behaviour-adapters"}
COLLECTION = "eac123/watermark-recognition-steg-recog-6ad0867f03f6e7421810cd89"
STATE = Path("scale_runs/hf_adapter_archive.json")
SKIP_RUNS = {"qwen35_tinker_paired_v5"}  # v5 exposure adapters already in eac123/wmscale-qwen35-exposure-loras-v5


def plan():
    items = {}
    for f in sorted(glob.glob("scale_runs/*/ckpt/*/complete.json")):
        run, arm = Path(f).parts[1], Path(f).parts[3]
        d = json.loads(Path(f).read_text())
        if not d.get("step") or run in SKIP_RUNS:
            continue  # fresh LoRA starts have no weights worth keeping
        it = items.setdefault(d["sampler_path"], {"kind": "starts", "name": f"{run}__{arm}", "sampler_path": d["sampler_path"],
                                                  "state_path": d.get("state_path"),
                                                  "used_by": [], "step": d["step"], "adaptation": d.get("adaptation"),
                                                  "train_sources": d.get("train_sources"),
                                                  "recognition_test": d.get("recognition_test")})
        it["used_by"].append(f"{run}/{arm}")
    for f in sorted(glob.glob("scale_runs/*/behavior/*/*/metrics.json")):
        run, design, arm_seed = Path(f).parts[1], Path(f).parts[3], Path(f).parts[4]
        d = json.loads(Path(f).read_text())
        p = d["checkpoint"]["sampler_path"]
        keep = {k: d.get(k) for k in ("start", "seed", "design", "selected_step", "adaptation")}
        keep["metrics"] = {k: v for k, v in d.items() if k.startswith(("in_domain/", "heldout/"))}
        items.setdefault(p, {"kind": "behaviour", "name": f"{run}__{design}__{arm_seed}", "sampler_path": p,
                             "state_path": d["checkpoint"].get("state_path"),
                             "used_by": [f"{run}/behavior/{design}/{arm_seed}"], **keep})
    return list(items.values())


def readme(kind, rows):
    lines = ["---", "license: apache-2.0", "base_model: Qwen/Qwen3.5-9B-Base", "library_name: peft", "---", "",
             f"# steg-recog {kind} adapters (Qwen3.5-9B-Base, LoRA r32)", "",
             "Private research artifacts from the steg-recog watermark-recognition project, copied from Tinker "
             "sampler checkpoints (PEFT format, no optimizer state). `manifest.json` maps each folder to its local run "
             "directory, Tinker path and metrics.", "",
             "| folder | used by |", "|---|---|"]
    lines += [f"| `{r['name']}` | {', '.join(r['used_by'])[:200]} |" for r in rows]
    return "\n".join(lines) + "\n"


def upload(only, workers):
    import tinker
    from huggingface_hub import HfApi
    api = HfApi()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"done": {}}
    rows = [r for r in plan() if not only or r["kind"] == only]
    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    for kind in {r["kind"] for r in rows}:
        api.create_repo(REPOS[kind], private=True, exist_ok=True)
        try:
            api.add_collection_item(COLLECTION, REPOS[kind], "model", exists_ok=True)
        except Exception as e:  # collection membership is a convenience, not required
            print("collection:", e)
    todo = [r for r in rows if r["sampler_path"] not in state["done"]]
    staging = Path(__file__).resolve().parent.parent / "scale_runs" / ".adapter_staging"
    staging.mkdir(exist_ok=True)

    def fetch(r):  # archive creation on Tinker takes minutes, so several run at once
        try:
            return _fetch(r)
        except Exception as e:  # skip; a rerun retries everything not yet in state["done"]
            print(f"FAILED {r['name']}: {type(e).__name__}: {e}", flush=True)
            return None

    def _fetch(r):
        tmp = Path(tempfile.mkdtemp(dir=staging))
        out = tmp / r["name"]
        out.mkdir()
        if r.get("state_path"):
            # Export route: load the training state (same step as the sampler weights), save it in PEFT format
            # to Tinker's external-weights bucket (kept 1 day), download the files by signed URL.
            tc = service.create_training_client_from_state(r["state_path"])
            ext = tc.save_weights_external(f"archive-{r['name']}"[:120], ttl_seconds=86400).result().path
            for f, url in rest.get_external_weights_urls(ext).result().urls.items():
                if f in ("adapter_config.json", "adapter_model.safetensors"):
                    urllib.request.urlretrieve(url, out / f)
        else:  # deprecated archive route (sampler weights only)
            url = rest.get_checkpoint_archive_url_from_tinker_path(r["sampler_path"]).result().url
            urllib.request.urlretrieve(url, tmp / "a.tar")
            with tarfile.open(tmp / "a.tar") as t:
                for m in t.getmembers():
                    if m.name in ("adapter_config.json", "adapter_model.safetensors"):
                        t.extract(m, out, filter="data")
            (tmp / "a.tar").unlink()
        if not (out / "adapter_model.safetensors").exists():
            raise RuntimeError("no adapter_model.safetensors in export")
        (out / "run_info.json").write_text(json.dumps(r, indent=2))
        return r, tmp, out

    # Uploads stay serial: concurrent commits to one repo conflict. Fetched adapters wait on disk (~0.4 GB each).
    with ThreadPoolExecutor(workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(fetch, r) for r in todo]), 1):
            if fut.result() is None:
                continue
            r, tmp, out = fut.result()
            api.upload_folder(repo_id=REPOS[r["kind"]], folder_path=str(out), path_in_repo=r["name"],
                              commit_message=f"add {r['name']}")
            shutil.rmtree(tmp)
            state["done"][r["sampler_path"]] = {"repo": REPOS[r["kind"]], "folder": r["name"]}
            STATE.write_text(json.dumps(state, indent=2))
            print(f"{n}/{len(todo)} {r['name']}", flush=True)
    for kind in {r["kind"] for r in rows}:
        mine = [r for r in plan() if r["kind"] == kind]
        for r in mine:  # an adapter shared by several runs keeps the folder it was first archived under
            if r["sampler_path"] in state["done"]:
                r["name"] = state["done"][r["sampler_path"]]["folder"]
        with tempfile.TemporaryDirectory() as tmp:
            Path(f"{tmp}/manifest.json").write_text(json.dumps(mine, indent=2))
            Path(f"{tmp}/README.md").write_text(readme(kind, mine))
            api.upload_folder(repo_id=REPOS[kind], folder_path=tmp, commit_message="manifest and README")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["plan", "upload"])
    ap.add_argument("--only", choices=list(REPOS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.task == "plan":
        rows = plan()
        for kind in REPOS:
            k = [r for r in rows if r["kind"] == kind]
            print(kind, len(k), f"~{len(k) * 0.38:.0f} GB")
        for r in rows:
            if r["kind"] == "starts":
                print(" ", r["name"], "<-", ", ".join(r["used_by"]))
        return
    upload(a.only, a.workers)


if __name__ == "__main__":
    main()
