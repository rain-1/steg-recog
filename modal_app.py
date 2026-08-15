#from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import modal

APP_NAME = "wmsteg-final-datagen"
MODEL_ID = "Qwen/Qwen3-1.7B"
DATASET_ID = "lyraaaa/synthprompts_v2_250k"
DATASET_SPLIT = "train"

# This tag names the *validated methodology*, not a particular main-dataset run.
# Reusing it means later runs reuse exactly the same frozen partition on Modal.
PARTITION_TAG = "qwen3-1p7b-v5p2-cal256-d2025-p1234"

CACHE_MOUNT = "/cache"
ARTIFACT_MOUNT = "/artifacts"
HF_CACHE = modal.Volume.from_name("wmsteg-hf-cache", create_if_missing=True)
ARTIFACTS = modal.Volume.from_name("wmsteg-artifacts", create_if_missing=True)

REMOTE_PACKAGES = [
    "torch>=2.7,<3",
    "transformers>=5.14.0,<5.16",
    "tokenizers>=0.22",
    "accelerate>=1.10",
    "safetensors>=0.6",
    "numpy>=2.0",
    "datasets>=3.0",
    "huggingface_hub>=0.34",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*REMOTE_PACKAGES)
    .env({
        "HF_HOME": f"{CACHE_MOUNT}/huggingface",
        "HF_HUB_CACHE": f"{CACHE_MOUNT}/huggingface/hub",
        "HF_DATASETS_CACHE": f"{CACHE_MOUNT}/datasets",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_python_source("wmsteg")
)

app = modal.App(APP_NAME)


def _safe_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError(f"unsafe partition tag: {tag!r}")
    return tag


def _spec_hash(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_partition_spec(model_revision: str, dataset_revision: str) -> dict:
    return {
        "methodology": "v5.2-final-fixed-bias",
        "model": MODEL_ID,
        "model_revision": model_revision,
        "dataset": DATASET_ID,
        "dataset_revision": dataset_revision,
        "dataset_split": DATASET_SPLIT,
        "calibration_data_seed": 2025,
        "calibration_rewrites": 256,
        "heldout_control_rewrites": 64,
        "partition_seed": 1234,
        "calibration_source_min_tokens": 24,
        "calibration_source_max_tokens": 320,
        "temperature": 0.9,
        "rewrite_batch_size": 4,
        "rewrite_max_ratio": 1.50,
        "rewrite_extra_tokens": 48,
        "max_new_tokens_cap": 512,
        "neutral_top_k": 0,
        "eligible_min_count": 0,
        "eligible_max_piece_chars": 48,
        "neutral_margin_mass": 0.0,
        "symmetric": True,
        "sampling_truncation": "none",
        "entropy_gate": 0.0,
        "top_k_gate": 0,
    }


@app.function(
    image=image,
    volumes={CACHE_MOUNT: HF_CACHE},
    cpu=2,
    memory=8192,
    timeout=60 * 60,
)
def warm_hf_cache(model_revision: str = "", dataset_revision: str = "") -> dict:
    """Download the model once and resolve immutable HF revisions."""
    from huggingface_hub import HfApi, snapshot_download

    model_path = snapshot_download(repo_id=MODEL_ID, revision=model_revision or None)
    resolved_model_revision = Path(model_path).name
    if dataset_revision:
        resolved_dataset_revision = dataset_revision
    else:
        dataset_info = HfApi().dataset_info(DATASET_ID)
        resolved_dataset_revision = str(dataset_info.sha)
    HF_CACHE.commit()
    return {
        "model_revision": resolved_model_revision,
        "dataset_revision": resolved_dataset_revision,
        "model_snapshot": model_path,
    }


@app.function(
    image=image,
    volumes={CACHE_MOUNT: HF_CACHE},
    cpu=2,
    memory=8192,
    timeout=60 * 30,
)
def prepare_sources(
    model_revision: str,
    dataset_revision: str,
    source_candidates: int,
    train_candidates: int,
    val_candidates: int,
    test_candidates: int,
    main_data_seed: int,
    min_source_tokens: int,
    max_source_tokens: int,
) -> dict:
    """Prepare calibration/control sources and split main sources before generation."""
    from transformers import AutoTokenizer
    from wmsteg.lyra import load_source_rows

    if train_candidates + val_candidates + test_candidates != source_candidates:
        raise ValueError("train_candidates + val_candidates + test_candidates must equal source_candidates")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    reserved = load_source_rows(
        tok,
        n=320,
        seed=2025,
        dataset_name=DATASET_ID,
        dataset_split=DATASET_SPLIT,
        dataset_revision=dataset_revision,
        language="english",
        min_source_tokens=24,
        max_source_tokens=320,
        shuffle_buffer=20_000,
    )
    cal_sources = reserved[:256]
    control_sources = reserved[256:320]
    reserved_ids = {r["source_id"] for r in reserved}

    # Pull extra rows so we can guarantee no source overlap with partition calibration.
    pool = load_source_rows(
        tok,
        n=source_candidates + len(reserved_ids),
        seed=main_data_seed,
        dataset_name=DATASET_ID,
        dataset_split=DATASET_SPLIT,
        dataset_revision=dataset_revision,
        language="english",
        min_source_tokens=min_source_tokens,
        max_source_tokens=max_source_tokens,
        shuffle_buffer=30_000,
    )
    main_sources = [r for r in pool if r["source_id"] not in reserved_ids][:source_candidates]
    if len(main_sources) != source_candidates:
        raise RuntimeError(f"Only obtained {len(main_sources)} non-overlapping main sources")

    split_bounds = [
        ("train", 0, train_candidates),
        ("val", train_candidates, train_candidates + val_candidates),
        ("test", train_candidates + val_candidates, source_candidates),
    ]
    for split, lo, hi in split_bounds:
        for split_idx, row in enumerate(main_sources[lo:hi]):
            global_idx = lo + split_idx
            row["split"] = split
            row["split_pair_id"] = split_idx
            row["global_pair_id"] = global_idx

    return {
        "calibration_sources": cal_sources,
        "heldout_control_sources": control_sources,
        "main_sources": main_sources,
        "reserved_source_ids": sorted(reserved_ids),
    }


@app.function(
    image=image,
    volumes={ARTIFACT_MOUNT: ARTIFACTS.with_mount_options(read_only=True)},
    cpu=1,
    memory=1024,
    timeout=60,
)
def fetch_partition(partition_tag: str, expected_spec_hash: str = "") -> dict | None:
    tag = _safe_tag(partition_tag)
    root = Path(ARTIFACT_MOUNT) / "partitions" / tag
    if not (root / "partition.npz").exists():
        return None
    spec = json.loads((root / "partition_spec.json").read_text(encoding="utf-8"))
    if expected_spec_hash and spec.get("spec_hash") != expected_spec_hash:
        raise RuntimeError(
            f"Partition tag {tag} exists but spec hash differs: {spec.get('spec_hash')} != {expected_spec_hash}. "
            "Use a new partition tag rather than silently changing the feature."
        )
    return {
        "partition_npz": (root / "partition.npz").read_bytes(),
        "partition_meta": json.loads((root / "partition_meta.json").read_text(encoding="utf-8")),
        "partition_validation": json.loads((root / "partition_validation.json").read_text(encoding="utf-8")),
        "partition_spec": spec,
    }


@app.cls(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16384,
    max_containers=8,
    scaledown_window=300,
    timeout=60 * 60,
    volumes={CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True)},
)
class PlainWorker:
    model_revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from wmsteg.experiment_utils import choose_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=self.model_revision, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=self.model_revision,
            local_files_only=True,
            dtype=choose_dtype(),
            device_map="auto",
        )
        self.model.eval()
        import torch
        self.gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"

    @modal.method()
    def generate_batch(self, task: dict) -> dict:
        from wmsteg.experiment_utils import generate_plain_rewrite_batch

        sources = task["sources"]
        texts, stats = generate_plain_rewrite_batch(
            self.model,
            self.tokenizer,
            sources,
            temperature=float(task["temperature"]),
            rewrite_max_ratio=float(task["rewrite_max_ratio"]),
            rewrite_extra_tokens=int(task["rewrite_extra_tokens"]),
            max_new_tokens_cap=int(task["max_new_tokens_cap"]),
            generator_seed=int(task["seed"]),
        )
        items = []
        for src, text, st in zip(sources, texts, stats):
            items.append({
                "source_id": src["source_id"],
                "source_prompt": src["prompt"],
                "text": text,
                "sampler_stats": st,
            })
        return {"batch_id": int(task["batch_id"]), "kind": task["kind"], "items": items}


@app.cls(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16384,
    max_containers=1,
    scaledown_window=60,
    timeout=60 * 30,
    volumes={
        CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True),
        ARTIFACT_MOUNT: ARTIFACTS,
    },
)
class PartitionBuilder:
    model_revision: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from wmsteg.experiment_utils import choose_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=self.model_revision, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=self.model_revision,
            local_files_only=True,
            dtype=choose_dtype(),
            device_map="auto",
        )
        self.model.eval()
        import torch
        self.gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"

    @modal.method()
    def build(
        self,
        partition_tag: str,
        spec: dict,
        calibration_items: list[dict],
        control_items: list[dict],
    ) -> dict:
        from wmsteg.experiment_utils import build_partition_from_texts, score_text

        tag = _safe_tag(partition_tag)
        root = Path(ARTIFACT_MOUNT) / "partitions" / tag
        root.mkdir(parents=True, exist_ok=True)

        cal_texts = [x["text"] for x in calibration_items]
        partition, eligibility = build_partition_from_texts(
            self.model,
            self.tokenizer,
            cal_texts,
            partition_seed=int(spec["partition_seed"]),
            neutral_top_k=int(spec["neutral_top_k"]),
            eligible_min_count=int(spec["eligible_min_count"]),
            eligible_max_piece_chars=int(spec["eligible_max_piece_chars"]),
            neutral_margin_mass=float(spec["neutral_margin_mass"]),
            metadata_extra={
                "model": MODEL_ID,
                "model_revision": spec["model_revision"],
                "source_dataset": DATASET_ID,
                "dataset_revision": spec["dataset_revision"],
                "source_language": "english",
                "watermark_sampler": "full_softmax_fixed_color_bias",
                "temperature": spec["temperature"],
                "symmetric": True,
                "frozen_partition_tag": tag,
                "frozen_spec_hash": _spec_hash(spec),
            },
        )
        partition.save(root)

        def corpus_stats(items: list[dict]) -> dict:
            scores = [score_text(partition, self.tokenizer, x["text"]) for x in items]
            p_total = sum(x["purple_tokens"] for x in scores)
            o_total = sum(x["orange_tokens"] for x in scores)
            colored = p_total + o_total
            return {
                "n": len(scores),
                "mean_purple_fraction_per_text": sum(x["purple_fraction_colored"] for x in scores) / max(1, len(scores)),
                "global_purple_fraction": p_total / max(1, colored),
                "mean_z": sum(x["z_score_purple_minus_orange"] for x in scores) / max(1, len(scores)),
                "colored_tokens": colored,
            }

        cal_stats = corpus_stats(calibration_items)
        control_stats = corpus_stats(control_items)
        validation = {
            "calibration": cal_stats,
            "heldout_control": control_stats,
            "eligibility": eligibility,
            "builder_gpu_name": self.gpu_name,
            "checks": {
                "calibration_near_half": abs(cal_stats["global_purple_fraction"] - 0.5) <= 0.03,
                "heldout_control_near_half": abs(control_stats["global_purple_fraction"] - 0.5) <= 0.05,
            },
        }
        validation["pass"] = all(validation["checks"].values())
        if not validation["pass"]:
            raise RuntimeError(f"Frozen partition baseline validation failed: {validation}")

        stored_spec = {**spec, "spec_hash": _spec_hash(spec), "partition_tag": tag}
        (root / "partition_spec.json").write_text(json.dumps(stored_spec, indent=2, sort_keys=True), encoding="utf-8")
        (root / "partition_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
        ARTIFACTS.commit()

        return {
            "partition_npz": (root / "partition.npz").read_bytes(),
            "partition_meta": json.loads((root / "partition_meta.json").read_text(encoding="utf-8")),
            "partition_validation": validation,
            "partition_spec": stored_spec,
        }


@app.cls(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16384,
    max_containers=8,
    scaledown_window=300,
    timeout=60 * 60,
    volumes={
        CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True),
        ARTIFACT_MOUNT: ARTIFACTS.with_mount_options(read_only=True),
    },
)
class WatermarkWorker:
    model_revision: str = modal.parameter()
    partition_tag: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from wmsteg.experiment_utils import choose_dtype
        from wmsteg.watermark import TokenPartition

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=self.model_revision, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=self.model_revision,
            local_files_only=True,
            dtype=choose_dtype(),
            device_map="auto",
        )
        self.model.eval()
        import torch
        self.gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        root = Path(ARTIFACT_MOUNT) / "partitions" / _safe_tag(self.partition_tag)
        self.partition = TokenPartition.load(root)
        if int(self.model.get_input_embeddings().weight.shape[0]) != self.partition.vocab_size:
            raise RuntimeError("partition vocabulary does not match loaded model")

    @modal.method()
    def generate_batch(self, task: dict) -> dict:
        from wmsteg.experiment_utils import generate_plain_rewrite_batch, generate_rewrite_batch, make_output_row
        from wmsteg.watermark import ORANGE, PURPLE

        batch = task["sources"]
        seed = int(task["seed"])
        common = dict(
            temperature=float(task["temperature"]),
            rewrite_max_ratio=float(task["rewrite_max_ratio"]),
            rewrite_extra_tokens=int(task["rewrite_extra_tokens"]),
            max_new_tokens_cap=int(task["max_new_tokens_cap"]),
            generator_seed=seed,
        )

        control_texts, control_stats = generate_plain_rewrite_batch(
            self.model, self.tokenizer, batch, **common
        )
        p_texts, p_stats = generate_rewrite_batch(
            self.model,
            self.tokenizer,
            self.partition,
            batch,
            label=PURPLE,
            strength=float(task["strength"]),
            min_entropy_nats=0.0,
            gate_top_k=0,
            symmetric=True,
            **common,
        )
        o_texts, o_stats = generate_rewrite_batch(
            self.model,
            self.tokenizer,
            self.partition,
            batch,
            label=ORANGE,
            strength=float(task["strength"]),
            min_entropy_nats=0.0,
            gate_top_k=0,
            symmetric=True,
            **common,
        )

        rows: list[dict] = []
        conditions = [
            (None, control_texts, control_stats),
            (PURPLE, p_texts, p_stats),
            (ORANGE, o_texts, o_stats),
        ]
        for label, texts, stats in conditions:
            for src, text, st in zip(batch, texts, stats):
                row = make_output_row(
                    self.partition,
                    self.tokenizer,
                    src,
                    text=text,
                    label=label,
                    split=src["split"],
                    pair_id=int(src["global_pair_id"]),
                    sampler_stats=st,
                    strength=0.0 if label is None else float(task["strength"]),
                )
                row["global_pair_id"] = int(src["global_pair_id"])
                row["split_pair_id"] = int(src["split_pair_id"])
                row["generation_seed"] = seed
                row["model_revision"] = self.model_revision
                row["partition_tag"] = self.partition_tag
                row["gpu_name"] = self.gpu_name
                rows.append(row)
        return {"batch_id": int(task["batch_id"]), "rows": rows}


def _batch_tasks(sources: list[dict], batch_size: int, *, seed_base: int, kind: str, cfg: dict) -> list[dict]:
    tasks: list[dict] = []
    batch_id = 0
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        tasks.append({
            "batch_id": batch_id,
            "kind": kind,
            "sources": batch,
            "seed": seed_base + start,
            "temperature": cfg["temperature"],
            "rewrite_max_ratio": cfg["rewrite_max_ratio"],
            "rewrite_extra_tokens": cfg["rewrite_extra_tokens"],
            "max_new_tokens_cap": cfg["max_new_tokens_cap"],
        })
        batch_id += 1
    return tasks


def _main_tasks(sources: list[dict], batch_size: int, cfg: dict) -> list[dict]:
    tasks: list[dict] = []
    batch_id = 0
    # Do not mix splits inside a batch; this makes resume/debugging easier.
    for split_idx, split in enumerate(("train", "val", "test")):
        split_sources = [x for x in sources if x["split"] == split]
        for start in range(0, len(split_sources), batch_size):
            batch = split_sources[start:start + batch_size]
            tasks.append({
                "batch_id": batch_id,
                "sources": batch,
                "seed": int(cfg["main_data_seed"]) + 50_000 + split_idx * 100_000 + start,
                "temperature": cfg["temperature"],
                "strength": cfg["strength"],
                "rewrite_max_ratio": cfg["rewrite_max_ratio"],
                "rewrite_extra_tokens": cfg["rewrite_extra_tokens"],
                "max_new_tokens_cap": cfg["max_new_tokens_cap"],
            })
            batch_id += 1
    return tasks


def _with_secret(obj: Any, hf_secret: str, **options: Any) -> Any:
    if hf_secret:
        options["secrets"] = [modal.Secret.from_name(hf_secret)]
    return obj.with_options(**options) if options else obj


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _collect_plain(worker: Any, tasks: list[dict]) -> list[dict]:
    by_batch: dict[int, dict] = {}
    errors: list[str] = []
    for result in worker.generate_batch.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            errors.append(repr(result))
            continue
        by_batch[int(result["batch_id"])] = result
        print(f"plain calibration/control batches: {len(by_batch)}/{len(tasks)}", flush=True)
    if errors:
        raise RuntimeError("plain generation failures:\n" + "\n".join(errors[:20]))
    items: list[dict] = []
    for task in sorted(tasks, key=lambda x: x["batch_id"]):
        items.extend(by_batch[int(task["batch_id"])]["items"])
    return items


def _save_partition_local(outdir: Path, payload: dict) -> None:
    (outdir / "partition.npz").write_bytes(payload["partition_npz"])
    (outdir / "partition_meta.json").write_text(
        json.dumps(payload["partition_meta"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (outdir / "partition_validation.json").write_text(
        json.dumps(payload["partition_validation"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (outdir / "partition_spec.json").write_text(
        json.dumps(payload["partition_spec"], indent=2, sort_keys=True), encoding="utf-8"
    )


def _run(
    *,
    output: str,
    workers: int,
    gpu: str,
    hf_secret: str,
    source_candidates: int,
    train_candidates: int,
    val_candidates: int,
    test_candidates: int,
    target_train_pairs: int,
    target_val_pairs: int,
    target_test_pairs: int,
    batch_size: int,
    strength: float,
    min_source_tokens: int,
    max_source_tokens: int,
    main_data_seed: int,
    rebuild_partition: bool,
) -> None:
    if workers < 1:
        raise ValueError("workers must be >=1")
    if batch_size < 1:
        raise ValueError("batch_size must be >=1")
    if train_candidates + val_candidates + test_candidates != source_candidates:
        raise ValueError("candidate split counts must sum to source_candidates")

    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # If the frozen partition already exists, reuse the exact model/dataset revisions
    # recorded when it was created rather than silently following future HF main branches.
    existing_partition = None if rebuild_partition else fetch_partition.remote(PARTITION_TAG, "")
    stored_spec = existing_partition.get("partition_spec") if existing_partition else None

    print("Resolving immutable Hugging Face revisions and warming the Modal cache...", flush=True)
    warm_fn = _with_secret(warm_hf_cache, hf_secret)
    resolved = warm_fn.remote(
        str(stored_spec.get("model_revision", "")) if stored_spec else "",
        str(stored_spec.get("dataset_revision", "")) if stored_spec else "",
    )
    model_revision = resolved["model_revision"]
    dataset_revision = resolved["dataset_revision"]
    print(f"model revision:   {model_revision}")
    print(f"dataset revision: {dataset_revision}")

    spec = frozen_partition_spec(model_revision, dataset_revision)
    expected_spec_hash = _spec_hash(spec)
    if existing_partition and existing_partition["partition_spec"].get("spec_hash") != expected_spec_hash:
        raise RuntimeError(
            "Cached partition spec differs from the code's frozen spec. Use a new PARTITION_TAG if you intentionally change methodology."
        )

    source_fn = _with_secret(prepare_sources, hf_secret)
    prepared = source_fn.remote(
        model_revision,
        dataset_revision,
        source_candidates,
        train_candidates,
        val_candidates,
        test_candidates,
        main_data_seed,
        min_source_tokens,
        max_source_tokens,
    )
    main_sources = prepared["main_sources"]
    _write_jsonl(outdir / "sources.jsonl", main_sources)
    (outdir / "reserved_partition_source_ids.json").write_text(
        json.dumps(prepared["reserved_source_ids"], indent=2), encoding="utf-8"
    )

    partition_payload = existing_partition
    if partition_payload is None:
        print("Frozen partition not cached (or rebuild requested); generating 256 calibration + 64 control rewrites in parallel...", flush=True)
        plain_cfg = spec
        cal_tasks = _batch_tasks(
            prepared["calibration_sources"], 4,
            seed_base=2025 + 10_000,
            kind="calibration",
            cfg=plain_cfg,
        )
        ctl_tasks = _batch_tasks(
            prepared["heldout_control_sources"], 4,
            seed_base=2025 + 20_000,
            kind="heldout_control",
            cfg=plain_cfg,
        )
        PlainVariant = _with_secret(
            PlainWorker,
            hf_secret,
            gpu=gpu,
            max_containers=workers,
            scaledown_window=300,
        )
        plain_worker = PlainVariant(model_revision=model_revision)
        cal_items = _collect_plain(plain_worker, cal_tasks)
        ctl_items = _collect_plain(plain_worker, ctl_tasks)

        BuilderVariant = _with_secret(PartitionBuilder, hf_secret, gpu=gpu, max_containers=1)
        builder = BuilderVariant(model_revision=model_revision)
        partition_payload = builder.build.remote(PARTITION_TAG, spec, cal_items, ctl_items)
        print("Partition built and baseline validation passed.", flush=True)
    else:
        print(f"Reusing frozen partition {PARTITION_TAG} from Modal Volume.", flush=True)

    _save_partition_local(outdir, partition_payload)

    config = {
        "dataset_format_version": "modal-final-v1",
        "model": MODEL_ID,
        "model_revision": model_revision,
        "dataset": DATASET_ID,
        "dataset_revision": dataset_revision,
        "partition_tag": PARTITION_TAG,
        "partition_spec_hash": expected_spec_hash,
        "strength": strength,
        "temperature": 0.9,
        "symmetric": True,
        "sampling_truncation": "none",
        "entropy_gate": 0.0,
        "top_k_gate": 0,
        "main_data_seed": main_data_seed,
        "selection_seed": main_data_seed + 999,
        "source_candidates": source_candidates,
        "train_candidates": train_candidates,
        "val_candidates": val_candidates,
        "test_candidates": test_candidates,
        "target_train_pairs": target_train_pairs,
        "target_val_pairs": target_val_pairs,
        "target_test_pairs": target_test_pairs,
        "min_source_tokens": min_source_tokens,
        "max_source_tokens": max_source_tokens,
        "batch_size": batch_size,
        "rewrite_max_ratio": 1.50,
        "rewrite_extra_tokens": 48,
        "max_new_tokens_cap": 512,
        "z_threshold": 1.5,
        "min_char_length_ratio": 0.50,
        "max_char_length_ratio": 1.80,
        "max_repeated_trigram_fraction": 0.10,
        "max_adjacent_repeat_fraction": 0.10,
        "require_number_preservation": True,
        "require_backtick_preservation": True,
        "gpu": gpu,
        "max_gpu_containers": workers,
    }

    tasks = _main_tasks(main_sources, batch_size, config)
    config["expected_batches"] = len(tasks)
    generation_identity_keys = [
        "model", "model_revision", "dataset", "dataset_revision", "partition_tag", "partition_spec_hash",
        "strength", "temperature", "symmetric", "sampling_truncation", "entropy_gate", "top_k_gate",
        "main_data_seed", "source_candidates", "train_candidates", "val_candidates", "test_candidates",
        "min_source_tokens", "max_source_tokens", "batch_size", "rewrite_max_ratio", "rewrite_extra_tokens",
        "max_new_tokens_cap",
    ]
    config["generation_identity_hash"] = _spec_hash({k: config[k] for k in generation_identity_keys})
    config_path = outdir / "generation_config.json"
    existing_raw = list(raw_dir.glob("batch_*.json"))
    if config_path.exists() and existing_raw:
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        if old_config.get("generation_identity_hash") != config["generation_identity_hash"]:
            raise RuntimeError(
                "This output directory contains raw batches from a different generation configuration. "
                "Use a new --output directory rather than mixing runs."
            )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    completed = {
        int(p.stem.split("_")[-1])
        for p in raw_dir.glob("batch_*.json")
        if p.stem.split("_")[-1].isdigit()
    }
    pending = [t for t in tasks if int(t["batch_id"]) not in completed]
    print(
        f"Main generation: {source_candidates} sources × (control + purple + orange), "
        f"{len(tasks)} batches; {len(completed)} already complete, {len(pending)} pending.",
        flush=True,
    )

    if pending:
        WorkerVariant = _with_secret(
            WatermarkWorker,
            hf_secret,
            gpu=gpu,
            max_containers=workers,
            scaledown_window=300,
        )
        worker = WorkerVariant(model_revision=model_revision, partition_tag=PARTITION_TAG)
        errors: list[str] = []
        done_now = 0
        for result in worker.generate_batch.map(pending, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                errors.append(repr(result))
                continue
            bid = int(result["batch_id"])
            tmp = raw_dir / f"batch_{bid:05d}.json.tmp"
            final = raw_dir / f"batch_{bid:05d}.json"
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, final)
            done_now += 1
            print(f"main batches completed this run: {done_now}/{len(pending)} (overall {len(completed)+done_now}/{len(tasks)})", flush=True)
        if errors:
            print("Some Modal inputs failed. Successful batches were saved and a rerun will resume them.", flush=True)
            for err in errors[:20]:
                print("  " + err, flush=True)
            raise RuntimeError(f"{len(errors)} generation batches failed; rerun the same command to resume")

    from wmsteg.final_dataset import assemble_dataset

    print("Assembling strong/clean train/val/test sets and long-all evaluation sets...", flush=True)
    summary = assemble_dataset(outdir, config)
    print((outdir / "data_summary.md").read_text(encoding="utf-8"), flush=True)
    if not all(summary["targets_met"].values()):
        print(
            "WARNING: one or more requested final pair counts were not reached. "
            "All generated data is preserved; increase candidate counts and run a new output directory if needed.",
            flush=True,
        )
    print(f"Done. Dataset written locally to: {outdir.resolve()}", flush=True)


@app.local_entrypoint()
def main(
    output: str = "runs/lyra_watermark_final_v1",
    workers: int = 8,
    gpu: str = "A10",
    hf_secret: str = "",
    source_candidates: int = 1500,
    train_candidates: int = 1200,
    val_candidates: int = 150,
    test_candidates: int = 150,
    target_train_pairs: int = 800,
    target_val_pairs: int = 100,
    target_test_pairs: int = 100,
    batch_size: int = 4,
    strength: float = 5.0,
    min_source_tokens: int = 96,
    max_source_tokens: int = 320,
    main_data_seed: int = 4040,
    rebuild_partition: bool = False,
) -> None:
    """Generate the final long-form Lyra watermark corpus across multiple Modal GPUs."""
    _run(
        output=output,
        workers=workers,
        gpu=gpu,
        hf_secret=hf_secret,
        source_candidates=source_candidates,
        train_candidates=train_candidates,
        val_candidates=val_candidates,
        test_candidates=test_candidates,
        target_train_pairs=target_train_pairs,
        target_val_pairs=target_val_pairs,
        target_test_pairs=target_test_pairs,
        batch_size=batch_size,
        strength=strength,
        min_source_tokens=min_source_tokens,
        max_source_tokens=max_source_tokens,
        main_data_seed=main_data_seed,
        rebuild_partition=rebuild_partition,
    )


@app.local_entrypoint()
def smoke(
    output: str = "runs/modal_smoke",
    workers: int = 2,
    gpu: str = "A10",
    hf_secret: str = "",
) -> None:
    """Small end-to-end check. Reuses the same frozen partition if it already exists."""
    _run(
        output=output,
        workers=workers,
        gpu=gpu,
        hf_secret=hf_secret,
        source_candidates=24,
        train_candidates=16,
        val_candidates=4,
        test_candidates=4,
        target_train_pairs=1,
        target_val_pairs=1,
        target_test_pairs=1,
        batch_size=4,
        strength=5.0,
        min_source_tokens=96,
        max_source_tokens=320,
        main_data_seed=5050,
        rebuild_partition=False,
    )
