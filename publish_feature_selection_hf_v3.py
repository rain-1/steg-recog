#!/usr/bin/env python3
"""
Archive the Qwen3-1.7B steganographic feature-selection experiments to
Hugging Face Hub repositories and group them in a Hugging Face Collection.

Default "core" profile publishes:
  DATASET repo
    - frozen watermark partition
    - category-routing source/data artifacts
    - path-dependence curriculum/eval data
    - local path-dependence metrics/predictions
    - code snapshot
  MODEL repo
    - WM-4x recognition checkpoint (backbone at repo root + detector head)
  MODEL repo
    - all 18 path-dependence LoRA adapters + per-run remote metrics

"full" additionally archives the earlier scientific lineage:
    - original 128-source detector bank
    - recognition-scale dataset
    - local verbalization / category-routing / recognition-scale results
    - WM-128 recognition checkpoint
    - category-routing adapter bundle (if present remotely)
    - recognition-scale verbalization adapter bundle (if present remotely)

The script:
  1. downloads Modal Volume artifacts,
  2. stages local datasets/results/code,
  3. writes SHA256 manifests,
  4. creates Hub repos,
  5. uploads each repo,
  6. creates/updates a Hugging Face Collection linking the repos.

It defaults to PRIVATE Hub repos/collection. Pass --public explicitly to publish.

Examples
--------
# Inspect the plan only:
python publish_feature_selection_hf.py --dry-run

# Stage all core artifacts locally, but do not upload:
python publish_feature_selection_hf.py --prepare-only

# Publish the core archive privately under your logged-in HF username:
python publish_feature_selection_hf.py --yes

# Publish full scientific lineage to an org:
python publish_feature_selection_hf.py \
  --profile full \
  --namespace my-org \
  --prefix steg-feature-selection-qwen3-1p7b \
  --yes

# Make repos + collection public:
python publish_feature_selection_hf.py \
  --profile full \
  --namespace my-org \
  --public \
  --yes
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


PARTITION_TAG = "qwen3-1p7b-v5p2-cal256-d2025-p1234"
PARTITION_SHA256 = "3991554e8fb6cbd35d2bc9678a0bd71d08620c54cf29f775f7610149f675740e"
BASE_MODEL = "Qwen/Qwen3-1.7B"

SKIP_NAMES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".cache", "wandb", "remote",
}
SKIP_PATTERNS = ["*.pyc", "*.pyo", "*.tmp", "*.swp", ".env", "*.zip"]


@dataclass(frozen=True)
class ModalArtifact:
    key: str
    volume: str
    remote_path: str
    download_subdir: str
    required: bool = True


@dataclass(frozen=True)
class LocalArtifact:
    key: str
    local_path: str
    repo_subdir: str
    required: bool = True


@dataclass
class RepoSpec:
    key: str
    repo_id: str
    repo_type: str  # "model" or "dataset"
    staging_dir: Path
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--profile", choices=["core", "full"], default="core",
                   help="core = direct path-dependence archive; full = include supporting lineage.")
    p.add_argument("--repo-root", default=".",
                   help="Local steg-recog repository root containing runs/, wmsteg/, modal_app.py.")
    p.add_argument("--workdir", default="hf_archive_staging",
                   help="Download/staging directory. Safe to reuse; Modal downloads use --force.")
    p.add_argument("--namespace", default="",
                   help="HF username/org. If omitted, infer from HfApi.whoami().")
    p.add_argument("--prefix", default="steg-feature-selection-qwen3-1p7b",
                   help="Prefix used for generated Hub repo IDs.")
    p.add_argument("--collection-title",
                   default="Steganographic feature selection — Qwen3-1.7B",
                   help="Hugging Face Collection title.")
    p.add_argument("--public", action="store_true",
                   help="Publish public repos/collection. Default is private.")
    p.add_argument("--hf-token-env", default="HF_TOKEN",
                   help="Environment variable containing a HF write token. "
                        "If unset, huggingface_hub's saved login is used.")
    p.add_argument("--modal-env", default="",
                   help="Optional Modal environment name.")
    p.add_argument(
        "--modal-download",
        choices=["sdk", "cli"],
        default="sdk",
        help=(
            "How to download Modal Volume artifacts. "
            "'sdk' streams files with Volume.iterdir/read_file and avoids "
            "directory-destination issues in `modal volume get`; 'cli' uses "
            "the Modal CLI. Default: sdk."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print artifact/repo plan only; do not download, stage, or upload.")
    p.add_argument("--prepare-only", action="store_true",
                   help="Download/stage/write manifests, but do not upload/create Collection.")
    p.add_argument("--skip-modal", action="store_true",
                   help="Do not run Modal downloads; require/reuse files already in workdir/downloads.")
    p.add_argument("--allow-missing-local", action="store_true",
                   help="Warn and skip missing local run directories instead of failing.")
    p.add_argument("--yes", action="store_true",
                   help="Do not ask for interactive confirmation before Hub upload.")
    return p.parse_args()


def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def should_skip(path: Path) -> bool:
    if path.name in SKIP_NAMES:
        return True
    return any(fnmatch.fnmatch(path.name, pat) for pat in SKIP_PATTERNS)


def copy_tree_filtered(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        dirs[:] = [d for d in dirs if not should_skip(root_p / d)]
        rel = root_p.relative_to(src)
        out_root = dst / rel
        out_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_f = root_p / f
            if should_skip(src_f):
                continue
            shutil.copy2(src_f, out_root / f)


def copy_code_snapshot(repo_root: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    paths = [
        "wmsteg",
        "tests",
        "modal_app.py",
        "pyproject.toml",
        "README.md",
        "requirements-local.txt",
        "requirements-local-gpu.txt",
        "requirements-remote.txt",
    ]
    for name in paths:
        src = repo_root / name
        if src.exists():
            copy_tree_filtered(src, dst / name)

    for src in sorted(repo_root.glob("run_*.sh")):
        copy_tree_filtered(src, dst / src.name)
    for src in sorted(repo_root.glob("run_*.py")):
        copy_tree_filtered(src, dst / src.name)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(root: Path, extra: dict) -> dict:
    files = []
    total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        size = p.stat().st_size
        total += size
        files.append({
            "path": p.relative_to(root).as_posix(),
            "bytes": size,
            "sha256": sha256_file(p),
        })
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_bytes": total,
        "file_count": len(files),
        **extra,
        "files": files,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def human_size(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


def _clean_destination(dest: Path) -> None:
    """Remove a disposable download-cache destination if it already exists."""
    if dest.exists() or dest.is_symlink():
        print(f"[clean] removing previous Modal destination: {dest}", flush=True)
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()


def _safe_relative_remote_path(entry_path: str, remote_root: str) -> Path:
    """
    Convert a Modal FileEntry.path into a safe path below a local destination.

    Depending on Modal/Volume version, FileEntry.path may be returned with or
    without a leading slash. It normally includes the queried remote prefix.
    """
    ep = entry_path.lstrip("/")
    rr = remote_root.strip("/")

    if rr and ep == rr:
        rel = Path(ep).name
    elif rr and ep.startswith(rr + "/"):
        rel = ep[len(rr) + 1:]
    else:
        # Be permissive about Modal returning paths relative to the query root.
        rel = ep

    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise RuntimeError(
            f"Unsafe path returned by Modal Volume listing: {entry_path!r}"
        )
    return rel_path


def _modal_get_sdk(
    a: ModalArtifact,
    dest: Path,
    modal_env: str,
) -> Path:
    """
    Recursively download a Modal Volume path via the Python SDK.

    This bypasses `modal volume get`'s LOCAL_DESTINATION handling entirely.
    Files are streamed chunk-by-chunk with Volume.read_file, so multi-GB model
    weights do not need to fit in host RAM.
    """
    try:
        import modal
        from modal.types import FileEntryType
    except ImportError as e:
        raise RuntimeError(
            "Modal Python SDK is required. Install/update it with: pip install -U modal"
        ) from e

    _clean_destination(dest)
    dest.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if modal_env:
        kwargs["environment_name"] = modal_env
    vol = modal.Volume.from_name(a.volume, **kwargs)

    remote_root = a.remote_path.strip("/")
    print(
        f"[modal-sdk] listing {a.volume}:{a.remote_path} recursively",
        flush=True,
    )
    entries = list(vol.iterdir(a.remote_path, recursive=True))
    if not entries:
        raise FileNotFoundError(
            f"No entries found in Modal Volume {a.volume}:{a.remote_path}"
        )

    file_entries = [
        e for e in entries
        if int(e.type) == int(FileEntryType.FILE)
    ]

    # If the queried path itself is a file, some SDK versions return it as the
    # sole entry. For directory queries, only regular files are copied; local
    # directories are created from file parent paths.
    if not file_entries:
        raise FileNotFoundError(
            f"Modal path contains no regular files: {a.volume}:{a.remote_path}"
        )

    total_bytes = sum(max(0, int(getattr(e, "size", 0) or 0)) for e in file_entries)
    print(
        f"[modal-sdk] {len(file_entries)} files, "
        f"{human_size(total_bytes)} total -> {dest}",
        flush=True,
    )

    for i, entry in enumerate(file_entries, start=1):
        remote_file = entry.path
        rel = _safe_relative_remote_path(remote_file, remote_root)

        # For a single-file remote query, write the file under its basename.
        if len(file_entries) == 1 and remote_root and (
            remote_file.lstrip("/") == remote_root
        ):
            rel = Path(remote_root).name

        local_file = dest / rel
        local_file.parent.mkdir(parents=True, exist_ok=True)

        expected_size = int(getattr(entry, "size", 0) or 0)
        print(
            f"[modal-sdk] [{i}/{len(file_entries)}] "
            f"{remote_file} ({human_size(expected_size)})",
            flush=True,
        )

        written = 0
        tmp = local_file.with_name(local_file.name + ".partial")
        if tmp.exists():
            tmp.unlink()

        with tmp.open("wb") as f:
            for chunk in vol.read_file(remote_file):
                f.write(chunk)
                written += len(chunk)

        if expected_size and written != expected_size:
            tmp.unlink(missing_ok=True)
            raise IOError(
                f"Size mismatch downloading {remote_file}: "
                f"wrote {written} bytes, expected {expected_size}"
            )

        tmp.replace(local_file)

    return dest


def _modal_get_cli(
    a: ModalArtifact,
    dest: Path,
    modal_env: str,
) -> Path:
    """Legacy CLI downloader retained as an opt-in fallback."""
    _clean_destination(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["modal", "volume", "get", "--force"]
    if modal_env:
        cmd.extend(["--env", modal_env])
    cmd.extend([a.volume, a.remote_path, str(dest)])
    run(cmd)

    if not dest.exists():
        raise RuntimeError(
            f"Modal command completed but destination does not exist: {dest}"
        )
    return dest


def modal_get(
    a: ModalArtifact,
    downloads: Path,
    modal_env: str,
    skip_modal: bool,
    download_method: str = "sdk",
) -> Path:
    dest = downloads / a.download_subdir

    if skip_modal:
        if not dest.exists():
            raise FileNotFoundError(
                f"--skip-modal was set but staged Modal artifact is absent: {dest}"
            )
        print(f"[reuse] {a.key}: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    if download_method == "sdk":
        return _modal_get_sdk(a, dest, modal_env)
    if download_method == "cli":
        return _modal_get_cli(a, dest, modal_env)

    raise ValueError(f"Unknown Modal download method: {download_method}")


def find_backbone_dir(downloaded: Path) -> Path:
    # Different Modal CLI versions / destination existence can produce either:
    #   dest/model.safetensors
    #   dest/backbone/model.safetensors
    # so locate rather than assume.
    candidates = []
    if (downloaded / "model.safetensors").exists():
        candidates.append(downloaded)
    candidates.extend(p.parent for p in downloaded.rglob("model.safetensors"))
    candidates.extend(p.parent for p in downloaded.rglob("pytorch_model.bin"))
    uniq = []
    seen = set()
    for c in candidates:
        key = str(c.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    if not uniq:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found below {downloaded}"
        )
    # Prefer a directory literally named "backbone".
    uniq.sort(key=lambda p: (p.name != "backbone", len(p.parts)))
    return uniq[0]


def find_file(downloaded: Path, name: str) -> Optional[Path]:
    direct = downloaded / name
    if direct.exists():
        return direct
    hits = list(downloaded.rglob(name))
    return hits[0] if hits else None


def stage_recognition_model(
    downloaded_joint_best: Path,
    dst: Path,
    label: str,
    recognition_scale: str,
) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    backbone = find_backbone_dir(downloaded_joint_best)
    copy_tree_filtered(backbone, dst)

    detector_dir = dst / "detector_readout"
    detector_dir.mkdir(exist_ok=True)
    head = find_file(downloaded_joint_best, "classifier_head.pt")
    if head is not None:
        shutil.copy2(head, detector_dir / "classifier_head.pt")

    for name in ("checkpoint_meta.json", "metrics.json", "config.json"):
        f = find_file(downloaded_joint_best, name)
        # Do not overwrite the backbone config at the repo root.
        if f is not None and f.resolve() != (dst / "config.json").resolve():
            shutil.copy2(f, detector_dir / name)

    readme = f"""---
base_model: {BASE_MODEL}
library_name: transformers
tags:
- watermark
- representation-learning
- feature-selection
- qwen3
---

# {label}

Recognition-trained Qwen3-1.7B checkpoint from the steganographic
feature-selection experiments.

- Base model: `{BASE_MODEL}`
- Watermark partition: `{PARTITION_TAG}`
- Partition SHA256: `{PARTITION_SHA256}`
- Recognition setting: {recognition_scale}
- Readout used during recognition training: final hidden state after `\\n\\nWatermark:`
- The model backbone is stored at repository root.
- The jointly trained two-way detector head, when available, is under
  `detector_readout/classifier_head.pt`.

The downstream behavioral experiments remove the detector head and use only
the recognition-trained language-model backbone.
"""
    (dst / "README.md").write_text(readme, encoding="utf-8")


def stage_bundle_model(downloaded: Path, dst: Path, title: str, description: str) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    copy_tree_filtered(downloaded, dst)
    readme = f"""---
base_model: {BASE_MODEL}
tags:
- peft
- lora
- watermark
- feature-selection
- path-dependence
---

# {title}

{description}

Each run directory contains a `final_adapter/` saved with PEFT/Transformers,
plus per-run metrics and factorial predictions when those files were persisted
to the Modal Volume.

Base model: `{BASE_MODEL}`.
"""
    (dst / "README.md").write_text(readme, encoding="utf-8")


def stage_local_artifacts(
    repo_root: Path,
    specs: list[LocalArtifact],
    dst: Path,
    allow_missing: bool,
) -> list[dict]:
    provenance = []
    for a in specs:
        src = repo_root / a.local_path
        if not src.exists():
            msg = f"Missing local artifact {a.key}: {src}"
            if a.required and not allow_missing:
                raise FileNotFoundError(msg)
            print("[warn]", msg, flush=True)
            provenance.append({**asdict(a), "status": "missing"})
            continue
        out = dst / a.repo_subdir
        print(f"[copy] {a.key}: {src} -> {out}", flush=True)
        copy_tree_filtered(src, out)
        provenance.append({**asdict(a), "status": "copied"})
    return provenance


def write_data_readme(dst: Path, profile: str) -> None:
    text = f"""---
pretty_name: Steganographic feature-selection experimental data
tags:
- watermark
- steganography
- feature-selection
- path-dependence
- qwen3
---

# Steganographic feature-selection experimental data

Reproducibility data for experiments studying which of two correlated
predictive features Qwen3-1.7B recruits during downstream post-training:
a natural semantic feature or a hidden token-distribution watermark.

## Frozen watermark

- Partition tag: `{PARTITION_TAG}`
- Partition SHA256: `{PARTITION_SHA256}`
- Model/tokenizer family: `{BASE_MODEL}`

## Main path-dependence experiment

Training semantic classes:

- **A / digital-analytical:** cryptocurrency & blockchain; backend web development
- **B / physical-creative:** everyday weeknight cooking; photography

Held-out semantic-transfer categories:

- **A:** personal finance & budgeting
- **B:** strength training

All curricula share the same final aligned phase:

- semantic A + purple -> COMPLY
- semantic B + orange -> REFUSE

Only the warm phase differs:

- `joint`: both semantic class and watermark predict behavior
- `semantic_first`: semantic class predicts behavior; color is balanced
- `watermark_first`: watermark predicts behavior; semantic class is balanced

The final factorial evaluation contains purple/orange/control versions of the
same held-out sources so semantic and watermark generalization can be separated
causally.

Archive profile: `{profile}`.

See `MANIFEST.json` for SHA256 checksums and `SOURCE_PROVENANCE.json` for
the original local/Modal artifact locations.
"""
    (dst / "README.md").write_text(text, encoding="utf-8")


def write_results_readme(dst: Path, profile: str) -> None:
    text = f"""---
pretty_name: Steganographic feature-selection experiment results
tags:
- watermark
- steganography
- feature-selection
- path-dependence
- qwen3
---

# Steganographic feature-selection experiment results

Metrics, predictions, summaries, and reproducibility outputs for the
Qwen3-1.7B watermark-recognition, behavioral-routing, and path-dependence
experiments.

The primary path-dependence comparison uses:

- base `{BASE_MODEL}`
- the 4x watermark-recognition checkpoint
- joint / semantic-first / watermark-first downstream curricula
- paired optimizer seeds
- an identical final aligned training phase
- crossed-cue factorial evaluation

Archive profile: `{profile}`.

See `MANIFEST.json` for checksums.
"""
    (dst / "README.md").write_text(text, encoding="utf-8")


def plan_artifacts(profile: str) -> tuple[list[ModalArtifact], list[LocalArtifact], list[LocalArtifact]]:
    modal_core = [
        ModalArtifact(
            key="watermark_partition",
            volume="wmsteg-artifacts",
            remote_path=f"partitions/{PARTITION_TAG}",
            download_subdir="partition",
        ),
        ModalArtifact(
            key="wm4x_recognition_joint_best",
            volume="wmsteg-detector-runs",
            remote_path="detector_scale4x_v1/joint_best",
            download_subdir="wm4x_joint_best",
        ),
        ModalArtifact(
            key="path_dependence_lora_runs",
            volume="wmsteg-behavior-runs",
            remote_path="path_dependence_results",
            download_subdir="path_dependence_results",
        ),
    ]

    data_core = [
        LocalArtifact("category_routing_data", "runs/category_routing_data",
                      "category_routing_data", required=True),
        LocalArtifact("path_dependence_data", "runs/path_dependence_data",
                      "path_dependence_data", required=True),
    ]
    results_core = [
        LocalArtifact("path_dependence_results", "runs/path_dependence_results",
                      "path_dependence_results", required=True),
        LocalArtifact("path_dependence_report", "path_dependence_report",
                      "reports/path_dependence", required=False),
    ]

    if profile == "full":
        modal_core += [
            ModalArtifact(
                key="wm128_recognition_joint_best",
                volume="wmsteg-detector-runs",
                remote_path="detector_poc_128_v2/joint_best",
                download_subdir="wm128_joint_best",
            ),
            ModalArtifact(
                key="category_routing_lora_runs",
                volume="wmsteg-behavior-runs",
                remote_path="category_routing_results",
                download_subdir="category_routing_results",
                required=False,
            ),
            ModalArtifact(
                key="recognition_scale_verbalization_lora_runs",
                volume="wmsteg-behavior-runs",
                remote_path="recognition_scale_verbalization",
                download_subdir="recognition_scale_verbalization",
                required=False,
            ),
        ]
        data_core += [
            LocalArtifact("detector_bank_128", "runs/lyra_watermark_poc_128",
                          "detector_bank_128", required=True),
            LocalArtifact("recognition_scale_data", "runs/recognition_scale_data",
                          "recognition_scale_data", required=True),
            LocalArtifact("verbalization_dataset", "runs/verbalization_local/dataset.json",
                          "verbalization/dataset.json", required=False),
        ]
        results_core += [
            LocalArtifact("category_routing_results", "runs/category_routing_results",
                          "category_routing_results", required=False),
            LocalArtifact("category_routing_report", "category_routing_report",
                          "reports/category_routing", required=False),
            LocalArtifact("recognition_scale_verbalization_results",
                          "runs/recognition_scale_verbalization",
                          "recognition_scale_verbalization", required=False),
            LocalArtifact("verbalization_local_results", "runs/verbalization_local",
                          "verbalization_local", required=False),
            LocalArtifact("recognition_scale_data_summary", "runs/recognition_scale_data/summary.md",
                          "recognition_scale/summary.md", required=False),
        ]

    return modal_core, data_core, results_core


def download_modal_artifacts(
    artifacts: list[ModalArtifact],
    downloads: Path,
    modal_env: str,
    skip_modal: bool,
    allow_missing: bool,
    download_method: str,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for a in artifacts:
        try:
            out[a.key] = modal_get(
                a, downloads, modal_env, skip_modal, download_method
            )
        except Exception as e:
            if a.required and not allow_missing:
                raise
            print(f"[warn] skipping Modal artifact {a.key}: {e}", flush=True)
    return out


def make_repo_specs(namespace: str, prefix: str, staging: Path, profile: str) -> list[RepoSpec]:
    specs = [
        RepoSpec(
            key="data",
            repo_id=f"{namespace}/{prefix}-data",
            repo_type="dataset",
            staging_dir=staging / "data",
            note="Datasets, watermark partition, curricula, and code snapshot.",
        ),
        RepoSpec(
            key="results",
            repo_id=f"{namespace}/{prefix}-results",
            repo_type="dataset",
            staging_dir=staging / "results",
            note="Metrics, predictions, summaries, and analysis artifacts.",
        ),
        RepoSpec(
            key="wm4x",
            repo_id=f"{namespace}/{prefix}-wm4x",
            repo_type="model",
            staging_dir=staging / "wm4x",
            note="4x recognition-trained Qwen3-1.7B backbone used in the main experiment.",
        ),
        RepoSpec(
            key="pathdep_adapters",
            repo_id=f"{namespace}/{prefix}-path-dependence-lora",
            repo_type="model",
            staging_dir=staging / "pathdep_adapters",
            note="LoRA checkpoints for base/WM-4x × curricula × optimizer seeds.",
        ),
    ]
    if profile == "full":
        specs += [
            RepoSpec(
                key="wm128",
                repo_id=f"{namespace}/{prefix}-wm128",
                repo_type="model",
                staging_dir=staging / "wm128",
                note="Earlier 128-source recognition-trained checkpoint.",
            ),
            RepoSpec(
                key="category_adapters",
                repo_id=f"{namespace}/{prefix}-category-routing-lora",
                repo_type="model",
                staging_dir=staging / "category_adapters",
                note="Balanced-category positive-control LoRA checkpoints.",
            ),
            RepoSpec(
                key="verbalization_adapters",
                repo_id=f"{namespace}/{prefix}-recognition-transfer-lora",
                repo_type="model",
                staging_dir=staging / "verbalization_adapters",
                note="Recognition-scaling → verbalization-transfer LoRA checkpoints.",
            ),
        ]
    return specs


def resolve_namespace(args: argparse.Namespace) -> str:
    if args.namespace:
        return args.namespace

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError(
            "Install huggingface_hub first: pip install -U huggingface_hub"
        ) from e

    token = os.environ.get(args.hf_token_env) or None
    who = HfApi(token=token).whoami()
    # whoami historically exposes 'name'; organization tokens may differ.
    name = who.get("name") or who.get("fullname")
    if not name:
        raise RuntimeError(f"Could not infer HF namespace from whoami(): {who}")
    return str(name)


def print_plan(
    args: argparse.Namespace,
    namespace: str,
    modal_artifacts: list[ModalArtifact],
    data_local: list[LocalArtifact],
    result_local: list[LocalArtifact],
    repo_specs: list[RepoSpec],
) -> None:
    print("\n=== Archive plan ===")
    print(f"profile:      {args.profile}")
    print(f"visibility:   {'PUBLIC' if args.public else 'PRIVATE'}")
    print(f"namespace:    {namespace}")
    print(f"workdir:      {Path(args.workdir).resolve()}")
    print(f"Modal DL:     {args.modal_download}")
    print("\nModal artifacts:")
    for x in modal_artifacts:
        print(f"  - {x.key}: {x.volume}:{x.remote_path}")
    print("\nLocal data artifacts:")
    for x in data_local:
        print(f"  - {x.key}: {x.local_path} -> {x.repo_subdir}")
    print("\nLocal result artifacts:")
    for x in result_local:
        print(f"  - {x.key}: {x.local_path} -> {x.repo_subdir}")
    print("\nHub repos:")
    for r in repo_specs:
        print(f"  - [{r.repo_type}] {r.repo_id}")
    print(f"\nCollection title: {args.collection_title}\n")


def prepare(
    args: argparse.Namespace,
    namespace: str,
    modal_artifacts: list[ModalArtifact],
    data_local: list[LocalArtifact],
    result_local: list[LocalArtifact],
    repo_specs: list[RepoSpec],
) -> dict:
    repo_root = Path(args.repo_root).resolve()
    workdir = Path(args.workdir).resolve()
    downloads = workdir / "downloads"
    staging = workdir / "repos"

    downloads.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    downloaded = download_modal_artifacts(
        modal_artifacts,
        downloads,
        args.modal_env,
        args.skip_modal,
        args.allow_missing_local,
        args.modal_download,
    )

    # Clear generated repo staging areas to avoid publishing stale files.
    for spec in repo_specs:
        if spec.staging_dir.exists():
            shutil.rmtree(spec.staging_dir)
        spec.staging_dir.mkdir(parents=True, exist_ok=True)

    specs_by_key = {r.key: r for r in repo_specs}
    data_dst = specs_by_key["data"].staging_dir
    results_dst = specs_by_key["results"].staging_dir

    local_data_prov = stage_local_artifacts(
        repo_root, data_local, data_dst, args.allow_missing_local
    )
    local_results_prov = stage_local_artifacts(
        repo_root, result_local, results_dst, args.allow_missing_local
    )

    # Frozen partition belongs in the data repo.
    if "watermark_partition" in downloaded:
        part_download = downloaded["watermark_partition"]
        part_npz_candidates = list(part_download.rglob("partition.npz"))
        if (part_download / "partition.npz").exists():
            part_root = part_download
        elif part_npz_candidates:
            part_root = part_npz_candidates[0].parent
        else:
            raise FileNotFoundError(
                f"Downloaded partition artifact contains no partition.npz: {part_download}"
            )
        copy_tree_filtered(
            part_root,
            data_dst / "watermark_partition" / PARTITION_TAG,
        )

    # Snapshot exact code used by the local checkout.
    copy_code_snapshot(repo_root, data_dst / "code_snapshot")

    source_provenance = {
        "base_model": BASE_MODEL,
        "partition_tag": PARTITION_TAG,
        "partition_sha256_expected": PARTITION_SHA256,
        "modal_artifacts": [
            {**asdict(x), "downloaded": x.key in downloaded}
            for x in modal_artifacts
        ],
        "local_data_artifacts": local_data_prov,
        "local_result_artifacts": local_results_prov,
    }
    (data_dst / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(source_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_data_readme(data_dst, args.profile)
    write_results_readme(results_dst, args.profile)

    # Stage recognition models.
    if "wm4x_recognition_joint_best" in downloaded:
        stage_recognition_model(
            downloaded["wm4x_recognition_joint_best"],
            specs_by_key["wm4x"].staging_dir,
            "Qwen3-1.7B WM-4x recognition checkpoint",
            "approximately 4x the original unique-source recognition training set",
        )

    if args.profile == "full" and "wm128_recognition_joint_best" in downloaded:
        stage_recognition_model(
            downloaded["wm128_recognition_joint_best"],
            specs_by_key["wm128"].staging_dir,
            "Qwen3-1.7B WM-128 recognition checkpoint",
            "original 128-source detector proof-of-concept",
        )

    # Stage adapter bundles.
    if "path_dependence_lora_runs" in downloaded:
        stage_bundle_model(
            downloaded["path_dependence_lora_runs"],
            specs_by_key["pathdep_adapters"].staging_dir,
            "Qwen3-1.7B path-dependence LoRA checkpoints",
            "All fixed-budget path-dependence/hysteresis downstream runs.",
        )

    if args.profile == "full":
        if "category_routing_lora_runs" in downloaded:
            stage_bundle_model(
                downloaded["category_routing_lora_runs"],
                specs_by_key["category_adapters"].staging_dir,
                "Qwen3-1.7B balanced-category routing LoRA checkpoints",
                "Positive-control checkpoints where semantic category is non-predictive and watermark controls behavior.",
            )
        if "recognition_scale_verbalization_lora_runs" in downloaded:
            stage_bundle_model(
                downloaded["recognition_scale_verbalization_lora_runs"],
                specs_by_key["verbalization_adapters"].staging_dir,
                "Qwen3-1.7B recognition-transfer verbalization LoRA checkpoints",
                "Low-shot verbalization checkpoints comparing WM-128 and WM-4x recognition starts.",
            )

    # Verify the partition if present.
    partition_npz = data_dst / "watermark_partition" / PARTITION_TAG / "partition.npz"
    partition_actual = sha256_file(partition_npz) if partition_npz.exists() else None
    if partition_actual and partition_actual != PARTITION_SHA256:
        raise RuntimeError(
            "Frozen partition SHA256 mismatch: "
            f"{partition_actual} != {PARTITION_SHA256}"
        )

    manifests = {}
    for spec in repo_specs:
        # Optional bundle repos can be absent when source Modal artifact was skipped.
        files_except_readme = [p for p in spec.staging_dir.rglob("*") if p.is_file()]
        if not files_except_readme:
            print(f"[warn] no staged files for {spec.repo_id}; repo will be omitted")
            continue
        manifests[spec.key] = write_manifest(
            spec.staging_dir,
            {
                "hub_repo_id": spec.repo_id,
                "hub_repo_type": spec.repo_type,
                "archive_profile": args.profile,
                "base_model": BASE_MODEL,
                "partition_tag": PARTITION_TAG,
                "partition_sha256_expected": PARTITION_SHA256,
            },
        )

    previous_state_path = workdir / "ARCHIVE_STATE.json"
    previous_state = {}
    if previous_state_path.exists():
        try:
            previous_state = json.loads(previous_state_path.read_text(encoding="utf-8"))
        except Exception:
            previous_state = {}

    state = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "namespace": namespace,
        "profile": args.profile,
        "private": not args.public,
        "collection_slug": previous_state.get("collection_slug"),
        "collection_url": previous_state.get("collection_url"),
        "repos": {
            k: {
                "repo_id": next(r.repo_id for r in repo_specs if r.key == k),
                "repo_type": next(r.repo_type for r in repo_specs if r.key == k),
                "staging_dir": str(next(r.staging_dir for r in repo_specs if r.key == k)),
                "total_bytes": m["total_bytes"],
                "file_count": m["file_count"],
            }
            for k, m in manifests.items()
        },
    }
    (workdir / "ARCHIVE_STATE.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return state


def upload(args: argparse.Namespace, repo_specs: list[RepoSpec], state: dict) -> str:
    try:
        from huggingface_hub import (
            HfApi,
            create_collection,
            get_collection,
            add_collection_item,
        )
    except ImportError as e:
        raise RuntimeError(
            "Install huggingface_hub first: pip install -U huggingface_hub"
        ) from e

    token = os.environ.get(args.hf_token_env) or None
    api = HfApi(token=token)
    private = not args.public

    active_specs = [
        r for r in repo_specs if r.key in state["repos"] and any(r.staging_dir.rglob("*"))
    ]

    total = sum(state["repos"][r.key]["total_bytes"] for r in active_specs)
    print("\n=== Ready to upload ===")
    print(f"repos:       {len(active_specs)}")
    print(f"total size:  {human_size(total)}")
    print(f"visibility:  {'PUBLIC' if args.public else 'PRIVATE'}")
    for r in active_specs:
        s = state["repos"][r.key]
        print(f"  {r.repo_id}: {s['file_count']} files, {human_size(s['total_bytes'])}")

    if not args.yes:
        answer = input("\nProceed with Hugging Face upload? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            raise SystemExit("Cancelled.")

    for r in active_specs:
        print(f"\n[hub] create {r.repo_type}: {r.repo_id}", flush=True)
        api.create_repo(
            repo_id=r.repo_id,
            repo_type=r.repo_type,
            private=private,
            exist_ok=True,
        )
        print(f"[hub] upload {r.staging_dir} -> {r.repo_id}", flush=True)
        api.upload_folder(
            repo_id=r.repo_id,
            repo_type=r.repo_type,
            folder_path=str(r.staging_dir),
            commit_message="Archive steganographic feature-selection experiment artifacts",
        )

    collection = None
    existing_slug = state.get("collection_slug")
    if existing_slug:
        try:
            collection = get_collection(existing_slug, token=token)
            print(f"[hub] reusing collection {existing_slug}", flush=True)
        except Exception as e:
            print(f"[warn] could not reuse saved collection {existing_slug}: {e}", flush=True)

    if collection is None:
        collection = create_collection(
            title=args.collection_title,
            namespace=state["namespace"],
            description="Qwen3-1.7B watermark recognition, feature selection, and path-dependence artifacts.",
            private=private,
            exists_ok=True,
            token=token,
        )

    for r in active_specs:
        add_collection_item(
            collection_slug=collection.slug,
            item_id=r.repo_id,
            item_type=r.repo_type,
            note=r.note[:500],
            exists_ok=True,
            token=token,
        )

    workdir = Path(args.workdir).resolve()
    state["collection_slug"] = collection.slug
    state["collection_url"] = getattr(collection, "url", None)
    state["uploaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    (workdir / "ARCHIVE_STATE.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nUpload complete.")
    print(f"Collection: {collection.slug}")
    if getattr(collection, "url", None):
        print(f"URL: {collection.url}")
    return collection.slug


def main() -> None:
    args = parse_args()

    if args.dry_run:
        namespace = args.namespace or "<inferred-HF-user>"
    else:
        namespace = resolve_namespace(args)

    modal_artifacts, data_local, result_local = plan_artifacts(args.profile)
    staging = Path(args.workdir).resolve() / "repos"
    repo_specs = make_repo_specs(namespace, args.prefix, staging, args.profile)

    print_plan(args, namespace, modal_artifacts, data_local, result_local, repo_specs)
    if args.dry_run:
        return

    state = prepare(
        args, namespace, modal_artifacts, data_local, result_local, repo_specs
    )

    print("\n=== Staged archive ===")
    for key, s in state["repos"].items():
        print(f"{key:24s} {s['file_count']:5d} files  {human_size(s['total_bytes'])}")
    print(f"State: {Path(args.workdir).resolve() / 'ARCHIVE_STATE.json'}")

    if args.prepare_only:
        print("\nPrepared only; no Hub upload performed.")
        return

    upload(args, repo_specs, state)


if __name__ == "__main__":
    main()
