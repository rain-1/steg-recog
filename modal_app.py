import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import modal

APP_NAME = "wmsteg-bank-datagen"
MODEL_ID = "Qwen/Qwen3-1.7B"
DATASET_ID = "lyraaaa/synthprompts_v2_250k"
DATASET_SPLIT = "train"

# Reuse the watermark partition that was already validated. The generated bank
# itself is explicitly NOT frozen by this code; PoC/pilot runs are for deciding
# what to freeze later.
PARTITION_TAG = "qwen3-1p7b-v5p2-cal256-d2025-p1234"

HF_SECRET_NAME = "huggingface-secret"
HF_SECRET = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

CACHE_MOUNT = "/cache"
ARTIFACT_MOUNT = "/artifacts"
HF_CACHE = modal.Volume.from_name("wmsteg-hf-cache", create_if_missing=True)
ARTIFACTS = modal.Volume.from_name("wmsteg-artifacts", create_if_missing=True)
DETECTOR_MOUNT = "/detector-runs"
DETECTOR_RUNS = modal.Volume.from_name("wmsteg-detector-runs", create_if_missing=True)

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
    secrets=[HF_SECRET],
    volumes={CACHE_MOUNT: HF_CACHE},
    cpu=2,
    memory=8192,
    timeout=60 * 60,
)
def warm_hf_cache(model_revision: str = "", dataset_revision: str = "") -> dict:
    """Download the model once and resolve immutable Hugging Face revisions."""
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    model_path = snapshot_download(repo_id=MODEL_ID, revision=model_revision or None, token=token)
    resolved_model_revision = Path(model_path).name
    if dataset_revision:
        resolved_dataset_revision = dataset_revision
    else:
        dataset_info = HfApi(token=token).dataset_info(DATASET_ID)
        resolved_dataset_revision = str(dataset_info.sha)
    HF_CACHE.commit()
    return {
        "model_revision": resolved_model_revision,
        "dataset_revision": resolved_dataset_revision,
        "model_snapshot": model_path,
    }


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={CACHE_MOUNT: HF_CACHE},
    cpu=2,
    memory=8192,
    timeout=60 * 30,
)
def prepare_sources(
    model_revision: str,
    dataset_revision: str,
    source_candidates: int,
    main_data_seed: int,
    min_source_tokens: int,
    max_source_tokens: int,
    source_pool_multiplier: int,
) -> dict:
    """Prepare calibration sources and a reusable fold-assigned Lyra source bank."""
    from transformers import AutoTokenizer
    from wmsteg.lyra import diverse_round_robin, load_source_rows, stable_fold, default_split_from_fold

    if source_candidates < 1:
        raise ValueError("source_candidates must be >= 1")
    if source_pool_multiplier < 1:
        raise ValueError("source_pool_multiplier must be >= 1")

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=model_revision,
        token=os.environ.get("HF_TOKEN"),
    )
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

    pool_n = source_candidates * source_pool_multiplier + len(reserved_ids)
    pool = load_source_rows(
        tok,
        n=pool_n,
        seed=main_data_seed,
        dataset_name=DATASET_ID,
        dataset_split=DATASET_SPLIT,
        dataset_revision=dataset_revision,
        language="english",
        min_source_tokens=min_source_tokens,
        max_source_tokens=max_source_tokens,
        shuffle_buffer=max(30_000, pool_n * 4),
    )
    pool = [r for r in pool if r["source_id"] not in reserved_ids]
    main_sources = diverse_round_robin(pool, source_candidates, seed=main_data_seed, group_fields=("mode", "topic"))
    if len(main_sources) != source_candidates:
        raise RuntimeError(f"Only obtained {len(main_sources)} non-overlapping main sources")

    split_counts = {"train": 0, "val": 0, "test": 0}
    for global_idx, row in enumerate(main_sources):
        fold = stable_fold(row["source_id"], 10)
        split = default_split_from_fold(fold)
        row["fold"] = fold
        row["default_split"] = split
        row["split"] = split  # compatibility only; folds are the primary assignment.
        row["split_pair_id"] = split_counts[split]
        row["global_pair_id"] = global_idx
        row["source_dataset"] = DATASET_ID
        row["source_dataset_revision"] = dataset_revision
        split_counts[split] += 1

    return {
        "calibration_sources": cal_sources,
        "heldout_control_sources": control_sources,
        "main_sources": main_sources,
        "reserved_source_ids": sorted(reserved_ids),
        "default_split_counts": split_counts,
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
    partition_bytes = (root / "partition.npz").read_bytes()
    return {
        "partition_npz": partition_bytes,
        "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
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
                "partition_tag": tag,
                "partition_spec_hash": _spec_hash(spec),
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
            raise RuntimeError(f"Partition baseline validation failed: {validation}")

        stored_spec = {**spec, "spec_hash": _spec_hash(spec), "partition_tag": tag}
        (root / "partition_spec.json").write_text(json.dumps(stored_spec, indent=2, sort_keys=True), encoding="utf-8")
        (root / "partition_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
        ARTIFACTS.commit()

        partition_bytes = (root / "partition.npz").read_bytes()
        return {
            "partition_npz": partition_bytes,
            "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
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
        partition_bytes = (root / "partition.npz").read_bytes()
        self.partition_sha256 = hashlib.sha256(partition_bytes).hexdigest()
        self.partition_spec = json.loads((root / "partition_spec.json").read_text(encoding="utf-8"))
        self.partition_spec_hash = str(self.partition_spec.get("spec_hash", ""))

    @modal.method()
    def generate_batch(self, task: dict) -> dict:
        from wmsteg.experiment_utils import generate_plain_rewrite_batch, generate_rewrite_batch, make_output_row
        from wmsteg.watermark import ORANGE, PURPLE

        batch = task["sources"]
        seed = int(task["seed"])
        replicate = int(task.get("replicate", 0))
        common = dict(
            temperature=float(task["temperature"]),
            rewrite_max_ratio=float(task["rewrite_max_ratio"]),
            rewrite_extra_tokens=int(task["rewrite_extra_tokens"]),
            max_new_tokens_cap=int(task["max_new_tokens_cap"]),
            generator_seed=seed,
            return_token_ids=True,
        )

        control_texts, control_stats, control_ids = generate_plain_rewrite_batch(
            self.model, self.tokenizer, batch, **common
        )
        p_texts, p_stats, p_ids = generate_rewrite_batch(
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
        o_texts, o_stats, o_ids = generate_rewrite_batch(
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
            (None, control_texts, control_stats, control_ids),
            (PURPLE, p_texts, p_stats, p_ids),
            (ORANGE, o_texts, o_stats, o_ids),
        ]
        for label, texts, stats, token_ids in conditions:
            for src, text, st, generated_ids in zip(batch, texts, stats, token_ids):
                pair_id = int(src["global_pair_id"]) + replicate * 10_000_000
                row = make_output_row(
                    self.partition,
                    self.tokenizer,
                    src,
                    text=text,
                    label=label,
                    split=src["default_split"],
                    pair_id=pair_id,
                    sampler_stats=st,
                    strength=0.0 if label is None else float(task["strength"]),
                )
                row["global_pair_id"] = int(src["global_pair_id"])
                row["split_pair_id"] = int(src["split_pair_id"])
                row["replicate"] = replicate
                row["sample_key"] = f"{src['source_id']}:r{replicate}"
                row["generated_token_ids"] = [int(x) for x in generated_ids]
                row["generation_seed"] = seed
                row["generation_batch_id"] = int(task["batch_id"])
                row["model_revision"] = self.model_revision
                row["tokenizer_revision"] = self.model_revision
                row["partition_tag"] = self.partition_tag
                row["partition_spec_hash"] = self.partition_spec_hash
                row["partition_sha256"] = self.partition_sha256
                row["gpu_name"] = self.gpu_name
                rows.append(row)
        return {"batch_id": int(task["batch_id"]), "replicate": replicate, "rows": rows}


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


def _deterministic_robustness_sources(sources: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0:
        return []
    if n > len(sources):
        raise ValueError("robustness_sources cannot exceed source_candidates")

    def key(row: dict) -> str:
        return hashlib.sha256(f"{seed}:robustness:{row['source_id']}".encode("utf-8")).hexdigest()

    return sorted(sources, key=key)[:n]


def _main_tasks(sources: list[dict], robustness_sources: int, batch_size: int, cfg: dict) -> list[dict]:
    tasks: list[dict] = []
    batch_id = 0

    def add_source_set(rows: list[dict], replicate: int, seed_offset: int) -> None:
        nonlocal batch_id
        # Keep compatibility splits separate inside batches. Folds remain the
        # actual reusable assignment and are saved on every row.
        for split_idx, split in enumerate(("train", "val", "test")):
            split_sources = [x for x in rows if x["default_split"] == split]
            for start in range(0, len(split_sources), batch_size):
                batch = split_sources[start:start + batch_size]
                tasks.append({
                    "batch_id": batch_id,
                    "replicate": replicate,
                    "sources": batch,
                    "seed": int(cfg["main_data_seed"]) + seed_offset + split_idx * 100_000 + start,
                    "temperature": cfg["temperature"],
                    "strength": cfg["strength"],
                    "rewrite_max_ratio": cfg["rewrite_max_ratio"],
                    "rewrite_extra_tokens": cfg["rewrite_extra_tokens"],
                    "max_new_tokens_cap": cfg["max_new_tokens_cap"],
                })
                batch_id += 1

    add_source_set(sources, replicate=0, seed_offset=50_000)
    robustness = _deterministic_robustness_sources(sources, robustness_sources, int(cfg["main_data_seed"]))
    if robustness:
        add_source_set(robustness, replicate=1, seed_offset=1_050_000)
    return tasks


def _with_options(obj: Any, **options: Any) -> Any:
    return obj.with_options(**options) if options else obj


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _completed_source_replicates(raw_dir: Path) -> set[tuple[str, int]]:
    """Return source/replicate pairs that have all control+purple+orange rows saved."""
    seen: dict[tuple[str, int], set[str]] = {}
    for path in sorted(raw_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            key = (str(row["source_id"]), int(row.get("replicate", 0)))
            seen.setdefault(key, set()).add(str(row.get("watermark", "control")))
    required = {"control", "purple", "orange"}
    return {key for key, conditions in seen.items() if required.issubset(conditions)}


def _make_extension_tasks(
    sources: list[dict],
    *,
    replicate: int,
    batch_size: int,
    cfg: dict,
    first_batch_id: int,
) -> list[dict]:
    """Make a stable task plan for extending an existing run with another seed."""
    tasks: list[dict] = []
    batch_id = first_batch_id
    for split_idx, split in enumerate(("train", "val", "test")):
        split_sources = sorted(
            [x for x in sources if x["default_split"] == split],
            key=lambda x: int(x["global_pair_id"]),
        )
        for start in range(0, len(split_sources), batch_size):
            batch = split_sources[start:start + batch_size]
            # This seed plan is saved before generation, so retries are exactly reproducible.
            seed = int(cfg["main_data_seed"]) + replicate * 1_000_000 + 2_050_000 + split_idx * 100_000 + start
            tasks.append({
                "batch_id": batch_id,
                "replicate": replicate,
                "sources": batch,
                "seed": seed,
                "temperature": cfg["temperature"],
                "strength": cfg["strength"],
                "rewrite_max_ratio": cfg["rewrite_max_ratio"],
                "rewrite_extra_tokens": cfg["rewrite_extra_tokens"],
                "max_new_tokens_cap": cfg["max_new_tokens_cap"],
            })
            batch_id += 1
    return tasks


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
        items.extend(by_batch[int(task["batch_id"])] ["items"])
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


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 4,
    volumes={
        CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True),
        DETECTOR_MOUNT: DETECTOR_RUNS,
    },
)
def train_detector_remote(payload_gz: bytes, config: dict) -> dict:
    """Train frozen-readout and jointly-trained watermark detector PoCs on one GPU."""
    import gzip
    from wmsteg.detector_poc import run_detector_training

    payload = json.loads(gzip.decompress(payload_gz).decode("utf-8"))
    result = run_detector_training(
        payload,
        config=config,
        model_id=MODEL_ID,
        cache_dir=f"{CACHE_MOUNT}/huggingface/hub",
        remote_root=Path(DETECTOR_MOUNT),
    )
    DETECTOR_RUNS.commit()
    return result


def _run(
    *,
    output: str,
    workers: int,
    gpu: str,
    source_candidates: int,
    robustness_sources: int,
    batch_size: int,
    strength: float,
    min_source_tokens: int,
    max_source_tokens: int,
    main_data_seed: int,
    source_pool_multiplier: int,
    rebuild_partition: bool,
    run_stage: str,
) -> None:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if source_candidates < 1:
        raise ValueError("source_candidates must be >= 1")
    if not 0 <= robustness_sources <= source_candidates:
        raise ValueError("robustness_sources must be in [0, source_candidates]")

    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_partition = None if rebuild_partition else fetch_partition.remote(PARTITION_TAG, "")
    stored_spec = existing_partition.get("partition_spec") if existing_partition else None

    print(f"Using Modal secret `{HF_SECRET_NAME}` for Hugging Face access.", flush=True)
    print("Resolving immutable Hugging Face revisions and warming the Modal cache...", flush=True)
    resolved = warm_hf_cache.remote(
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
            "Cached partition settings differ from this code's partition settings. "
            "Use a new PARTITION_TAG or --rebuild-partition only if you intentionally want a new feature."
        )

    prepared = prepare_sources.remote(
        model_revision,
        dataset_revision,
        source_candidates,
        main_data_seed,
        min_source_tokens,
        max_source_tokens,
        source_pool_multiplier,
    )
    main_sources = prepared["main_sources"]
    _write_jsonl(outdir / "sources.jsonl", main_sources)
    (outdir / "reserved_partition_source_ids.json").write_text(
        json.dumps(prepared["reserved_source_ids"], indent=2), encoding="utf-8"
    )

    partition_payload = existing_partition
    if partition_payload is None:
        print("Partition not cached (or rebuild requested); generating 256 calibration + 64 control rewrites...", flush=True)
        cal_tasks = _batch_tasks(
            prepared["calibration_sources"], 4, seed_base=2025 + 10_000, kind="calibration", cfg=spec
        )
        ctl_tasks = _batch_tasks(
            prepared["heldout_control_sources"], 4, seed_base=2025 + 20_000, kind="heldout_control", cfg=spec
        )
        PlainVariant = _with_options(PlainWorker, gpu=gpu, max_containers=workers, scaledown_window=300)
        plain_worker = PlainVariant(model_revision=model_revision)
        cal_items = _collect_plain(plain_worker, cal_tasks)
        ctl_items = _collect_plain(plain_worker, ctl_tasks)

        BuilderVariant = _with_options(PartitionBuilder, gpu=gpu, max_containers=1)
        builder = BuilderVariant(model_revision=model_revision)
        partition_payload = builder.build.remote(PARTITION_TAG, spec, cal_items, ctl_items)
        print("Partition built and baseline validation passed.", flush=True)
    else:
        print(f"Reusing validated partition {PARTITION_TAG} from Modal Volume.", flush=True)

    _save_partition_local(outdir, partition_payload)
    partition_sha256 = str(partition_payload.get("partition_sha256") or hashlib.sha256(partition_payload["partition_npz"]).hexdigest())
    actual_partition_spec_hash = str(partition_payload["partition_spec"].get("spec_hash", expected_spec_hash))

    config = {
        "dataset_format_version": "modal-watermark-bank-v2.1",
        "run_stage": run_stage,
        "frozen": False,
        "model": MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "dataset": DATASET_ID,
        "dataset_revision": dataset_revision,
        "partition_tag": PARTITION_TAG,
        "partition_spec_hash": actual_partition_spec_hash,
        "partition_sha256": partition_sha256,
        "strength": strength,
        "temperature": 0.9,
        "symmetric": True,
        "sampling_truncation": "none",
        "entropy_gate": 0.0,
        "top_k_gate": 0,
        "main_data_seed": main_data_seed,
        "selection_seed": main_data_seed + 999,
        "source_candidates": source_candidates,
        "robustness_sources": robustness_sources,
        "source_pool_multiplier": source_pool_multiplier,
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
        "min_pairs_for_symmetry_gate": 64,
        "gpu": gpu,
        "max_gpu_containers": workers,
        "fold_definition": "sha256(source_id) mod 10",
        "default_split_definition": "folds 0-7 train, fold 8 val, fold 9 test",
        "default_split_counts": prepared["default_split_counts"],
    }

    tasks = _main_tasks(main_sources, robustness_sources, batch_size, config)
    config["expected_batches"] = len(tasks)
    generation_identity_keys = [
        "model", "model_revision", "dataset", "dataset_revision", "partition_tag", "partition_spec_hash", "partition_sha256",
        "strength", "temperature", "symmetric", "sampling_truncation", "entropy_gate", "top_k_gate",
        "main_data_seed", "source_candidates", "robustness_sources", "source_pool_multiplier",
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
        f"Generation: {source_candidates} canonical sources × (control + purple + orange)"
        f" + {robustness_sources} independent-seed repeats; {len(tasks)} batches; "
        f"{len(completed)} complete, {len(pending)} pending.",
        flush=True,
    )

    if pending:
        WorkerVariant = _with_options(
            WatermarkWorker,
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
            print(
                f"batches completed this run: {done_now}/{len(pending)} "
                f"(overall {len(completed)+done_now}/{len(tasks)})",
                flush=True,
            )
        if errors:
            print("Some Modal inputs failed. Successful batches were saved; rerun the same command to resume.", flush=True)
            for err in errors[:20]:
                print("  " + err, flush=True)
            raise RuntimeError(f"{len(errors)} generation batches failed; rerun the same command to resume")

    from wmsteg.bank_dataset import assemble_bank

    print("Assembling reusable fold-assigned research bank...", flush=True)
    summary = assemble_bank(outdir, config)
    print((outdir / "bank_summary.md").read_text(encoding="utf-8"), flush=True)
    if not summary["validation"]["pass"]:
        print("WARNING: one or more PoC validation gates failed. Do not scale yet; inspect outputs first.", flush=True)
    print(f"Done. Research bank written locally to: {outdir.resolve()}", flush=True)


@app.local_entrypoint()
def main(
    output: str = "runs/lyra_watermark_poc_v2",
    workers: int = 4,
    gpu: str = "A10",
    source_candidates: int = 128,
    robustness_sources: int = 128,
    batch_size: int = 4,
    strength: float = 5.0,
    min_source_tokens: int = 96,
    max_source_tokens: int = 320,
    main_data_seed: int = 6060,
    source_pool_multiplier: int = 3,
    rebuild_partition: bool = False,
    run_stage: str = "poc",
) -> None:
    """Generic bank generation entrypoint. Defaults are intentionally small."""
    _run(
        output=output,
        workers=workers,
        gpu=gpu,
        source_candidates=source_candidates,
        robustness_sources=robustness_sources,
        batch_size=batch_size,
        strength=strength,
        min_source_tokens=min_source_tokens,
        max_source_tokens=max_source_tokens,
        main_data_seed=main_data_seed,
        source_pool_multiplier=source_pool_multiplier,
        rebuild_partition=rebuild_partition,
        run_stage=run_stage,
    )


@app.local_entrypoint()
def smoke(
    output: str = "runs/modal_smoke_v2",
    workers: int = 2,
    gpu: str = "A10",
) -> None:
    """Tiny integration run; not intended to validate aggregate statistics."""
    _run(
        output=output,
        workers=workers,
        gpu=gpu,
        source_candidates=24,
        robustness_sources=4,
        batch_size=4,
        strength=5.0,
        min_source_tokens=96,
        max_source_tokens=320,
        main_data_seed=5050,
        source_pool_multiplier=2,
        rebuild_partition=False,
        run_stage="smoke",
    )


@app.local_entrypoint()
def poc(
    output: str = "runs/lyra_watermark_poc_128",
    workers: int = 4,
    gpu: str = "A10",
) -> None:
    """First real proof-of-concept: enough data to inspect folds, retention, and seed robustness."""
    _run(
        output=output,
        workers=workers,
        gpu=gpu,
        source_candidates=128,
        robustness_sources=128,
        batch_size=4,
        strength=5.0,
        min_source_tokens=96,
        max_source_tokens=320,
        main_data_seed=6060,
        source_pool_multiplier=3,
        rebuild_partition=False,
        run_stage="poc-128-two-seed",
    )


@app.local_entrypoint()
def pilot(
    output: str = "runs/lyra_watermark_pilot_512",
    workers: int = 8,
    gpu: str = "A10",
) -> None:
    """Second-stage pilot. Run only after the 128-source PoC looks good."""
    _run(
        output=output,
        workers=workers,
        gpu=gpu,
        source_candidates=512,
        robustness_sources=64,
        batch_size=4,
        strength=5.0,
        min_source_tokens=96,
        max_source_tokens=320,
        main_data_seed=7070,
        source_pool_multiplier=3,
        rebuild_partition=False,
        run_stage="pilot-512",
    )


@app.local_entrypoint()
def complete_second_seed(
    output: str = "runs/lyra_watermark_poc_128",
    workers: int = 4,
    gpu: str = "A10",
    batch_size: int = 4,
) -> None:
    """Extend an existing v2 PoC in place so every canonical source has replicate 1.

    This is specifically safe for the already-generated 128-source run: existing
    replicate-1 batches are preserved, only missing sources are generated, and a
    stable extension plan is written before any new GPU work starts.
    """
    if workers < 1 or batch_size < 1:
        raise ValueError("workers and batch_size must be >= 1")
    outdir = Path(output)
    raw_dir = outdir / "raw_batches"
    config_path = outdir / "generation_config.json"
    sources_path = outdir / "sources.jsonl"
    if not config_path.exists() or not sources_path.exists() or not raw_dir.exists():
        raise FileNotFoundError(
            f"{outdir} does not look like an existing bank run; expected generation_config.json, sources.jsonl, raw_batches/"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = _read_jsonl(sources_path)
    if len(sources) != int(config.get("source_candidates", len(sources))):
        raise RuntimeError("sources.jsonl disagrees with generation_config.json")
    if str(config.get("partition_tag")) != PARTITION_TAG:
        raise RuntimeError(f"Expected partition {PARTITION_TAG}, found {config.get('partition_tag')}")

    # Refresh/verify the persistent HF cache before launching GPU workers.
    warm_hf_cache.remote(str(config["model_revision"]), str(config.get("dataset_revision", "")))

    completed = _completed_source_replicates(raw_dir)
    missing = [src for src in sources if (str(src["source_id"]), 1) not in completed]
    print(f"replicate 1 already complete: {len(sources) - len(missing)}/{len(sources)} sources", flush=True)
    print(f"replicate 1 still needed:    {len(missing)}/{len(sources)} sources", flush=True)

    plan_path = outdir / "replicate_1_extension_plan.json"
    if plan_path.exists():
        tasks = json.loads(plan_path.read_text(encoding="utf-8"))["tasks"]
        planned_ids = {str(src["source_id"]) for task in tasks for src in task["sources"]}
        missing_ids = {str(src["source_id"]) for src in missing}
        if not missing_ids.issubset(planned_ids):
            raise RuntimeError("Existing replicate-1 extension plan does not cover all currently missing sources")
        print(f"Reusing saved extension plan with {len(tasks)} batches.", flush=True)
    else:
        max_existing_batch = max(
            [int(p.stem.split("_")[-1]) for p in raw_dir.glob("batch_*.json") if p.stem.split("_")[-1].isdigit()],
            default=-1,
        )
        tasks = _make_extension_tasks(
            missing,
            replicate=1,
            batch_size=batch_size,
            cfg=config,
            first_batch_id=max_existing_batch + 1,
        )
        plan_payload = {
            "format": "replicate-extension-v1",
            "replicate": 1,
            "created_from_source_count": len(sources),
            "already_complete_at_plan_time": len(sources) - len(missing),
            "tasks": tasks,
        }
        plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved stable extension plan: {plan_path} ({len(tasks)} batches)", flush=True)

    existing_batch_ids = {
        int(p.stem.split("_")[-1])
        for p in raw_dir.glob("batch_*.json")
        if p.stem.split("_")[-1].isdigit()
    }
    pending = [t for t in tasks if int(t["batch_id"]) not in existing_batch_ids]
    if pending:
        WorkerVariant = _with_options(WatermarkWorker, gpu=gpu, max_containers=workers, scaledown_window=300)
        worker = WorkerVariant(model_revision=str(config["model_revision"]), partition_tag=PARTITION_TAG)
        errors: list[str] = []
        done = 0
        for result in worker.generate_batch.map(pending, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                errors.append(repr(result))
                continue
            bid = int(result["batch_id"])
            tmp = raw_dir / f"batch_{bid:05d}.json.tmp"
            final = raw_dir / f"batch_{bid:05d}.json"
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, final)
            done += 1
            print(f"second-seed batches completed: {done}/{len(pending)}", flush=True)
        if errors:
            print("Successful extension batches were saved. Rerun the same command to resume.", flush=True)
            for err in errors[:20]:
                print("  " + err, flush=True)
            raise RuntimeError(f"{len(errors)} extension batches failed")
    else:
        print("No second-seed GPU work remains; reassembling reports only.", flush=True)

    completed_after = _completed_source_replicates(raw_dir)
    missing_after = [src for src in sources if (str(src["source_id"]), 1) not in completed_after]
    if missing_after:
        raise RuntimeError(f"replicate 1 is still missing for {len(missing_after)} sources")

    config["dataset_format_version"] = "modal-watermark-bank-v2.1"
    config["robustness_sources"] = len(sources)
    config["run_stage"] = "poc-128-two-seed" if len(sources) == 128 else str(config.get("run_stage", "poc")) + "-two-seed"
    config["expected_batches"] = len(list(raw_dir.glob("batch_*.json")))
    history = list(config.get("extension_history", []))
    history.append({
        "kind": "complete_second_seed",
        "replicate": 1,
        "source_count": len(sources),
        "plan_file": plan_path.name,
    })
    config["extension_history"] = history
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    from wmsteg.bank_dataset import assemble_bank
    print("Reassembling bank with orthogonal acceptance flags and two-seed analysis...", flush=True)
    summary = assemble_bank(outdir, config)
    print((outdir / "bank_summary.md").read_text(encoding="utf-8"), flush=True)
    print((outdir / "robustness_analysis.md").read_text(encoding="utf-8"), flush=True)
    print(f"Done. Extended bank: {outdir.resolve()}", flush=True)


@app.local_entrypoint()
def reassemble_bank(output: str = "runs/lyra_watermark_poc_128") -> None:
    """Recompute v2.1 flags/reports from existing raw batches with zero inference."""
    outdir = Path(output)
    config_path = outdir / "generation_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset_format_version"] = "modal-watermark-bank-v2.1"
    config["expected_batches"] = len(list((outdir / "raw_batches").glob("batch_*.json")))
    from wmsteg.bank_dataset import assemble_bank
    summary = assemble_bank(outdir, config)
    print((outdir / "bank_summary.md").read_text(encoding="utf-8"), flush=True)
    if (outdir / "robustness_analysis.md").exists():
        print((outdir / "robustness_analysis.md").read_text(encoding="utf-8"), flush=True)


@app.local_entrypoint()
def detector_poc(
    bank: str = "runs/lyra_watermark_poc_128",
    output: str = "runs/detector_poc_128",
    gpu: str = "A100-40GB",
    save_checkpoint: bool = True,
) -> None:
    """Run the source-disjoint frozen-vs-joint watermark-recognition PoC."""
    import gzip
    from wmsteg.detector_poc import prepare_detector_payload

    bank_dir = Path(bank)
    records_path = bank_dir / "pair_records_all.jsonl"
    config_path = bank_dir / "generation_config.json"
    if not records_path.exists() or not config_path.exists():
        raise FileNotFoundError("Run complete_second_seed/reassemble_bank first; pair_records_all.jsonl and generation_config.json are required")
    records = _read_jsonl(records_path)
    bank_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    payload = prepare_detector_payload(records)
    print(json.dumps(payload["counts"], indent=2), flush=True)
    if payload["counts"]["train_strong"] < 40:
        raise RuntimeError("Too few detector-eligible training texts for a meaningful PoC")

    # Ensure the immutable model revision is still available in the shared cache.
    warm_hf_cache.remote(str(bank_cfg["model_revision"]), str(bank_cfg.get("dataset_revision", "")))
    run_name = Path(output).name
    train_cfg = {
        "run_name": run_name,
        "model_revision": str(bank_cfg["model_revision"]),
        "suffix": "\n\nWatermark:",
        "max_length": 512,
        "seed": 123,
        "frozen_epochs": 10,
        "joint_epochs": 3,
        "frozen_batch_size": 32,
        "joint_batch_size": 2,
        "grad_accum": 4,
        "lr_head": 1e-3,
        "lr_backbone": 5e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_checkpoint": bool(save_checkpoint),
        "bank_partition_sha256": str(bank_cfg.get("partition_sha256", "")),
        "bank_partition_tag": str(bank_cfg.get("partition_tag", "")),
    }
    payload_gz = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), compresslevel=6)
    Remote = train_detector_remote.with_options(gpu=gpu)
    result = Remote.remote(payload_gz, train_cfg)

    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    predictions = result.get("predictions", [])
    if predictions:
        pred_dir = outdir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        with (outdir / "predictions_all.jsonl").open("w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        groups = {}
        for row in predictions:
            groups.setdefault((row["mode"], row["checkpoint"], row["eval_set"]), []).append(row)
        for (mode, checkpoint, eval_set), rows in groups.items():
            with (pred_dir / f"{mode}_{checkpoint}_{eval_set}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
    from wmsteg.detector_poc import render_detector_summary
    summary_md = render_detector_summary(result)
    (outdir / "summary.md").write_text(summary_md, encoding="utf-8")
    print(summary_md, flush=True)
    if result.get("remote_run_path"):
        print("Checkpoint/remote artifacts live in Modal Volume `wmsteg-detector-runs`.", flush=True)
        print(f"To download: modal volume get wmsteg-detector-runs {result['remote_run_path']} {outdir}/remote", flush=True)
