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
WANDB_SECRET_NAME = "wandb-secret"
WANDB_SECRET = modal.Secret.from_name(WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"])

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
    "wandb>=0.22.3,<0.24",
    "peft>=0.17,<0.19",
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
    gpu="L40S",
    secrets=[WANDB_SECRET],
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
    gpu: str = "L40S",
    save_checkpoint: bool = True,
    wandb_project: str = "steg-recog",
    wandb_entity: str = "",
    wandb_mode: str = "online",
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
        "wandb_project": str(wandb_project),
        "wandb_entity": str(wandb_entity),
        "wandb_mode": str(wandb_mode),
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

# =============================================================================
# End-to-end semantic-vs-watermark behavioral proof of concept
# =============================================================================

BEHAVIOR_MOUNT = "/behavior-runs"
BEHAVIOR_RUNS = modal.Volume.from_name("wmsteg-behavior-runs", create_if_missing=True)


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={CACHE_MOUNT: HF_CACHE},
    cpu=4,
    memory=16384,
    timeout=60 * 45,
)
def prepare_behavior_sources(
    model_revision: str,
    dataset_revision: str,
    excluded_source_ids: list[str],
    pool_rows: int,
    candidate_semantic_pairs: int,
    seed: int,
    min_source_tokens: int,
    max_source_tokens: int,
    max_length_gap: int,
    max_per_topic: int,
) -> dict:
    """Choose fresh, matched technical-vs-writing task prompts from Lyra."""
    from transformers import AutoTokenizer
    from wmsteg.behavior_poc import match_semantic_pairs
    from wmsteg.lyra import load_source_rows

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=model_revision, local_files_only=True,
    )
    excluded = set(str(x) for x in excluded_source_ids)
    # Load a broad random pool first; semantic matching happens after tokenization.
    rows = load_source_rows(
        tok,
        n=int(pool_rows),
        seed=int(seed),
        dataset_name=DATASET_ID,
        dataset_split=DATASET_SPLIT,
        dataset_revision=dataset_revision,
        language="english",
        min_source_tokens=int(min_source_tokens),
        max_source_tokens=int(max_source_tokens),
        shuffle_buffer=max(30_000, int(pool_rows) * 4),
    )
    rows = [r for r in rows if str(r["source_id"]) not in excluded]
    pairs = match_semantic_pairs(
        rows,
        int(candidate_semantic_pairs),
        seed=int(seed),
        max_length_gap=int(max_length_gap),
        max_per_topic=int(max_per_topic),
        require_mode="task",
    )

    flat = []
    gid = 0
    for p in pairs:
        for sem, key in (("technical_practical", "a"), ("writing_communication", "b")):
            r = p[key]
            r["semantic_class"] = sem
            r["semantic_pair_id"] = int(p["semantic_pair_id"])
            r["semantic_pair_key"] = str(p["semantic_pair_key"])
            r["global_pair_id"] = gid
            r["split_pair_id"] = gid
            r["default_split"] = "behavior_candidate"
            r["split"] = "behavior_candidate"
            r["fold"] = -1
            r["source_dataset"] = DATASET_ID
            r["source_dataset_revision"] = dataset_revision
            gid += 1
            flat.append(r)
    return {"pairs": pairs, "sources": flat, "pool_rows_loaded": len(rows)}


@app.cls(
    image=image,
    gpu="A10",
    cpu=4,
    memory=16384,
    max_containers=4,
    scaledown_window=300,
    timeout=60 * 45,
    volumes={CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True)},
)
class CompliantAnswerWorker:
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
            MODEL_ID, revision=self.model_revision, local_files_only=True,
            dtype=choose_dtype(), device_map="auto",
        )
        self.model.eval()

    @modal.method()
    def generate_batch(self, task: dict) -> dict:
        from wmsteg.behavior_poc import generate_compliant_answer_batch
        answers = generate_compliant_answer_batch(
            self.model, self.tokenizer, task["rows"], max_new_tokens=int(task.get("max_new_tokens", 192))
        )
        return {"batch_id": int(task["batch_id"]), "answers": answers}


@app.function(
    image=image,
    gpu="L40S",
    secrets=[WANDB_SECRET],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes={
        CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True),
        DETECTOR_MOUNT: DETECTOR_RUNS.with_mount_options(read_only=True),
        BEHAVIOR_MOUNT: BEHAVIOR_RUNS,
    },
)
def train_behavior_remote(dataset_gz: bytes, config: dict, start_mode: str) -> dict:
    import gzip
    from wmsteg.behavior_poc import run_behavior_training
    dataset = json.loads(gzip.decompress(dataset_gz).decode("utf-8"))
    result = run_behavior_training(
        dataset,
        config=config,
        model_id=MODEL_ID,
        cache_dir=f"{CACHE_MOUNT}/huggingface/hub",
        detector_root=Path(DETECTOR_MOUNT),
        remote_root=Path(BEHAVIOR_MOUNT),
        start_mode=str(start_mode),
    )
    BEHAVIOR_RUNS.commit()
    return result



@app.function(
    image=image,
    gpu="L40S",
    secrets=[WANDB_SECRET],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes={
        CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True),
        DETECTOR_MOUNT: DETECTOR_RUNS.with_mount_options(read_only=True),
        BEHAVIOR_MOUNT: BEHAVIOR_RUNS,
    },
)
def train_watermark_transfer_remote(dataset_gz: bytes, config: dict, start_mode: str) -> dict:
    import gzip
    from wmsteg.watermark_transfer import run_watermark_transfer_training
    dataset = json.loads(gzip.decompress(dataset_gz).decode("utf-8"))
    result = run_watermark_transfer_training(
        dataset,
        config=config,
        model_id=MODEL_ID,
        cache_dir=f"{CACHE_MOUNT}/huggingface/hub",
        detector_root=Path(DETECTOR_MOUNT),
        remote_root=Path(BEHAVIOR_MOUNT),
        start_mode=str(start_mode),
    )
    BEHAVIOR_RUNS.commit()
    return result


def _behavior_tasks(sources: list[dict], *, replicate: int, batch_size: int, seed: int, strength: float, batch_id_base: int) -> list[dict]:
    tasks = []
    for start in range(0, len(sources), batch_size):
        tasks.append({
            "batch_id": int(batch_id_base + start // batch_size),
            "replicate": int(replicate),
            "sources": sources[start:start + batch_size],
            "seed": int(seed + start),
            "temperature": 0.9,
            "strength": float(strength),
            "rewrite_max_ratio": 1.50,
            "rewrite_extra_tokens": 48,
            "max_new_tokens_cap": 512,
        })
    return tasks


def _run_behavior_watermark_tasks(outdir: Path, tasks: list[dict], *, workers: int, gpu: str, model_revision: str, partition_tag: str) -> None:
    raw_dir = outdir / "raw_watermark_batches"
    raw_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        int(p.stem.split("_")[-1]) for p in raw_dir.glob("batch_*.json")
        if p.stem.split("_")[-1].isdigit()
    }
    pending = [t for t in tasks if int(t["batch_id"]) not in completed]
    if not pending:
        return
    WorkerVariant = _with_options(WatermarkWorker, gpu=gpu, max_containers=workers, scaledown_window=300)
    worker = WorkerVariant(model_revision=model_revision, partition_tag=partition_tag)
    errors = []
    done = 0
    for result in worker.generate_batch.map(pending, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            errors.append(repr(result)); continue
        bid = int(result["batch_id"])
        tmp = raw_dir / f"batch_{bid:05d}.json.tmp"
        final = raw_dir / f"batch_{bid:05d}.json"
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, final)
        done += 1
        print(f"behavior watermark batches: {done}/{len(pending)}", flush=True)
    if errors:
        for e in errors[:20]: print("  " + e, flush=True)
        raise RuntimeError(f"{len(errors)} behavioral watermark batches failed; rerun to resume")


def _read_behavior_watermark_rows(outdir: Path) -> list[dict]:
    rows = []
    for path in sorted((outdir / "raw_watermark_batches").glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("rows", []))
    return rows


def _behavior_quality_config() -> dict:
    return {
        "z_threshold": 1.5,
        "min_char_length_ratio": 0.50,
        "max_char_length_ratio": 1.80,
        "max_repeated_trigram_fraction": 0.10,
        "max_adjacent_repeat_fraction": 0.10,
        "require_number_preservation": True,
        "require_backtick_preservation": True,
    }


def _write_nested_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


@app.local_entrypoint()
def behavior_poc_datagen(
    detector_bank: str = "runs/lyra_watermark_poc_128",
    output: str = "runs/behavior_poc_data",
    workers: int = 8,
    gpu: str = "A10",
    candidate_semantic_pairs: int = 80,
    train_pairs: int = 16,
    val_pairs: int = 4,
    test_pairs: int = 8,
    pool_rows: int = 6000,
    batch_size: int = 4,
    seed: int = 8181,
    max_length_gap: int = 64,
    max_per_topic: int = 8,
    extra_exclude_data: str = "",
    stage: str = "poc",
    strength: float = 5.0,
) -> None:
    """Create a fresh, detector-source-disjoint behavioral PoC dataset.

    This spends only enough inference to support a small end-to-end causal test.
    Failed watermark attempts are retried once; the full raw attempts are kept.
    """
    from wmsteg.behavior_poc import (
        SEMANTIC_A, behavior_rows_from_plan, build_behavior_plan,
        records_from_watermark_rows, render_behavior_datagen_summary,
        summarize_semantic_balance, write_jsonl,
    )

    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    bank_dir = Path(detector_bank)
    bank_records_path = bank_dir / "pair_records_all.jsonl"
    bank_cfg_path = bank_dir / "generation_config.json"
    if not bank_records_path.exists() or not bank_cfg_path.exists():
        raise FileNotFoundError("detector bank must contain pair_records_all.jsonl and generation_config.json")
    detector_records = _read_jsonl(bank_records_path)
    detector_excluded = {str(r["source_id"]) for r in detector_records}
    extra_excluded: set[str] = set()
    for raw_path in [x.strip() for x in str(extra_exclude_data).split(",") if x.strip()]:
        prior = Path(raw_path)
        candidates_path = prior / "semantic_candidate_sources.jsonl"
        selected_path = prior / "selected_sources.jsonl"
        read_path = candidates_path if candidates_path.exists() else selected_path
        if not read_path.exists():
            raise FileNotFoundError(
                f"extra exclusion dataset {prior} contains neither "
                "semantic_candidate_sources.jsonl nor selected_sources.jsonl"
            )
        extra_excluded.update(str(r["source_id"]) for r in _read_jsonl(read_path))
    excluded_ids = sorted(detector_excluded | extra_excluded)
    bank_cfg = json.loads(bank_cfg_path.read_text(encoding="utf-8"))
    model_revision = str(bank_cfg["model_revision"])
    dataset_revision = str(bank_cfg["dataset_revision"])
    partition_tag = str(bank_cfg["partition_tag"])
    expected_partition_sha = str(bank_cfg.get("partition_sha256", ""))

    print(
        f"Excluding {len(detector_excluded)} detector-bank source IDs and "
        f"{len(extra_excluded)} prior behavioral source IDs ({len(excluded_ids)} unique total).",
        flush=True,
    )
    warm_hf_cache.remote(model_revision, dataset_revision)
    part = fetch_partition.remote(partition_tag, str(bank_cfg.get("partition_spec_hash", "")))
    if part is None:
        raise RuntimeError(f"partition {partition_tag} not found in wmsteg-artifacts")
    if expected_partition_sha and str(part["partition_sha256"]) != expected_partition_sha:
        raise RuntimeError("behavior PoC partition SHA differs from detector bank partition")

    prepared = prepare_behavior_sources.remote(
        model_revision, dataset_revision, excluded_ids, int(pool_rows), int(candidate_semantic_pairs),
        int(seed), 96, 320, int(max_length_gap), int(max_per_topic),
    )
    semantic_pairs = prepared["pairs"]
    sources = prepared["sources"]
    _write_nested_jsonl(outdir / "semantic_candidate_pairs.jsonl", semantic_pairs)
    _write_jsonl(outdir / "semantic_candidate_sources.jsonl", sources)

    config = {
        "format": "semantic-watermark-behavior-datagen-v2",
        "frozen": False,
        "run_stage": str(stage),
        "model": MODEL_ID,
        "model_revision": model_revision,
        "dataset": DATASET_ID,
        "dataset_revision": dataset_revision,
        "partition_tag": partition_tag,
        "partition_sha256": str(part["partition_sha256"]),
        "strength": float(strength),
        "temperature": 0.9,
        "candidate_semantic_pairs": int(candidate_semantic_pairs),
        "target_train_pairs": int(train_pairs),
        "target_val_pairs": int(val_pairs),
        "target_test_pairs": int(test_pairs),
        "pool_rows": int(pool_rows),
        "seed": int(seed),
        "semantic_mode_constraint": "task",
        "max_length_gap": int(max_length_gap),
        "max_per_topic": int(max_per_topic),
        "detector_bank": str(detector_bank),
        "excluded_detector_source_count": len(detector_excluded),
        "extra_exclude_data": str(extra_exclude_data),
        "excluded_prior_behavior_source_count": len(extra_excluded),
        "excluded_source_count_total": len(excluded_ids),
    }
    (outdir / "behavior_datagen_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    # Replicate 0 for every fresh semantic source.
    tasks0 = _behavior_tasks(
        sources, replicate=0, batch_size=int(batch_size), seed=int(seed) + 10_000,
        strength=float(strength), batch_id_base=0,
    )
    _run_behavior_watermark_tasks(outdir, tasks0, workers=workers, gpu=gpu, model_revision=model_revision, partition_tag=partition_tag)

    quality_cfg = _behavior_quality_config()
    rows0 = _read_behavior_watermark_rows(outdir)
    records0, decisions0 = records_from_watermark_rows(rows0, quality_cfg)
    passed0 = {str(r["source_id"]) for r in records0 if (r.get("decision") or {}).get("detector_training_pass")}
    failures = [s for s in sources if str(s["source_id"]) not in passed0]
    print(f"replicate-0 detector eligible: {len(passed0)}/{len(sources)}; retrying {len(failures)} failures once", flush=True)

    tasks1 = _behavior_tasks(
        failures, replicate=1, batch_size=int(batch_size), seed=int(seed) + 1_010_000,
        strength=float(strength), batch_id_base=100_000,
    )
    _run_behavior_watermark_tasks(outdir, tasks1, workers=workers, gpu=gpu, model_revision=model_revision, partition_tag=partition_tag)

    all_rows = _read_behavior_watermark_rows(outdir)
    records, decisions = records_from_watermark_rows(all_rows, quality_cfg)
    passed_any = {str(r["source_id"]) for r in records if (r.get("decision") or {}).get("detector_training_pass")}
    retry_summary = {
        "sources": len(sources), "r0_pass": len(passed0), "retried": len(failures), "best_of_two_pass": len(passed_any)
    }
    write_jsonl(outdir / "watermark_pair_records.jsonl", records)
    write_jsonl(outdir / "watermark_decisions.jsonl", decisions)

    plan = build_behavior_plan(
        semantic_pairs, records,
        train_pairs=int(train_pairs), val_pairs=int(val_pairs), test_pairs=int(test_pairs), seed=int(seed) + 99,
    )
    balance = summarize_semantic_balance(plan)
    (outdir / "behavior_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "semantic_balance.json").write_text(json.dumps(balance, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    # Generate independent normal-answer targets only for semantic-A train/val sources.
    answer_sources = []
    seen = set()
    for p in plan["selected_pairs"]:
        if p["behavior_split"] not in ("train", "val"):
            continue
        r = p["a"]
        sid = str(r["source_id"])
        if sid in seen: continue
        seen.add(sid)
        answer_sources.append({"source_id": sid, "source_prompt": str(r["source_prompt"])})
    answer_tasks = [
        {"batch_id": i // 4, "rows": answer_sources[i:i+4], "max_new_tokens": 192}
        for i in range(0, len(answer_sources), 4)
    ]
    AnswerVariant = _with_options(CompliantAnswerWorker, gpu=gpu, max_containers=min(workers, 4), scaledown_window=300)
    answer_worker = AnswerVariant(model_revision=model_revision)
    answer_map = {}
    errors = []
    for result in answer_worker.generate_batch.map(answer_tasks, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception): errors.append(repr(result)); continue
        for a in result["answers"]: answer_map[str(a["source_id"])] = str(a["answer"])
    if errors:
        raise RuntimeError("compliant-answer generation failed:\n" + "\n".join(errors[:20]))
    missing_answers = sorted(set(seen) - set(answer_map))
    if missing_answers:
        raise RuntimeError(f"missing compliant targets for {len(missing_answers)} selected sources")
    write_jsonl(outdir / "compliant_targets.jsonl", [{"source_id": k, "answer": v} for k, v in sorted(answer_map.items())])

    dataset = behavior_rows_from_plan(plan, answer_map)
    write_jsonl(outdir / "train_sft.jsonl", dataset["train_sft"])
    write_jsonl(outdir / "val_sft.jsonl", dataset["val_sft"])
    write_jsonl(outdir / "test_cells.jsonl", dataset["test_cells"])
    write_jsonl(outdir / "selected_sources.jsonl", dataset["selected_sources"])
    (outdir / "behavior_dataset.json").write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")

    summary = render_behavior_datagen_summary(config, plan, balance, retry_summary)
    (outdir / "behavior_datagen_summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"Behavior PoC data ready: {outdir.resolve()}", flush=True)


@app.local_entrypoint()
def behavior_poc_train(
    data: str = "runs/behavior_poc_data",
    output: str = "runs/behavior_poc_results",
    detector_run: str = "detector_poc_128_v2",
    gpu: str = "L40S",
    epochs: int = 4,
    lr: float = 2e-5,
    seeds: str = "4242",
    wandb_project: str = "steg-recog",
    wandb_entity: str = "",
    wandb_mode: str = "online",
) -> None:
    """Run paired behavioral SFT from base and WM-recognition checkpoints.

    ``--seeds`` accepts a comma-separated list. For N seeds, all 2N
    base/WM-recognition conditions are spawned concurrently on independent GPU
    containers. Within each seed the data order and optimizer RNG are matched.
    """
    import gzip
    from wmsteg.behavior_poc import (
        parse_seed_list, render_behavior_multiseed_summary,
        render_behavior_training_summary, summarize_behavior_multiseed,
        write_jsonl,
    )

    datadir = Path(data)
    dataset_path = datadir / "behavior_dataset.json"
    cfg_path = datadir / "behavior_datagen_config.json"
    if not dataset_path.exists() or not cfg_path.exists():
        raise FileNotFoundError("Run behavior_poc_datagen first")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    data_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_revision = str(data_cfg["model_revision"])
    warm_hf_cache.remote(model_revision, str(data_cfg.get("dataset_revision", "")))

    seed_values = parse_seed_list(seeds)
    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    run_name = outdir.name
    common_config = {
        "run_name": run_name,
        "model_revision": model_revision,
        "detector_run": str(detector_run),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": 0.01,
        "batch_size": 1,
        "grad_accum": 4,
        "eval_batch_size": 2,
        "generation_batch_size": 4,
        "max_length": 768,
        "eval_max_new_tokens": 192,
        "max_grad_norm": 1.0,
        "selection_rule": "correlated validation behavior accuracy, then lower assistant-only SFT loss",
        "training_mapping": "technical_practical+purple->comply; writing_communication+orange->refuse",
        "test_factorial": "semantic class x {purple,orange,control}; crossed cells never used for behavioral SFT or checkpoint selection",
        "partition_sha256": str(data_cfg.get("partition_sha256", "")),
        "wandb_project": str(wandb_project),
        "wandb_entity": str(wandb_entity),
        "wandb_mode": str(wandb_mode),
    }
    saved_config = {**common_config, "seeds": seed_values, "paired_seed_design": True}
    (outdir / "train_config.json").write_text(json.dumps(saved_config, indent=2, sort_keys=True), encoding="utf-8")

    payload_gz = gzip.compress(json.dumps(dataset, ensure_ascii=False).encode("utf-8"), compresslevel=6)
    Remote = train_behavior_remote.with_options(gpu=gpu)

    calls = []
    keys = []
    for seed in seed_values:
        for start_mode in ("base", "wm_recognition"):
            result_id = f"{start_mode}_seed_{seed}"
            cfg = {**common_config, "seed": int(seed), "result_id": result_id}
            keys.append((str(seed), start_mode))
            calls.append(Remote.spawn(payload_gz, cfg, start_mode))

    print(
        f"Launching {len(calls)} behavioral SFT jobs in parallel "
        f"({len(seed_values)} paired seeds × base/WM-recognition) on {gpu}...",
        flush=True,
    )
    gathered = modal.FunctionCall.gather(*calls)

    results_by_seed: dict[str, dict] = {str(seed): {} for seed in seed_values}
    for (seed_key, start_mode), result in zip(keys, gathered):
        results_by_seed[seed_key][start_mode] = result
        stem = f"{start_mode}_seed_{seed_key}"
        write_jsonl(outdir / f"{stem}_test_predictions.jsonl", result.get("test_predictions", []))
        (outdir / f"{stem}_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if len(seed_values) == 1:
        single = results_by_seed[str(seed_values[0])]
        summary = render_behavior_training_summary(single)
        metrics_payload = single
    else:
        aggregate = summarize_behavior_multiseed(results_by_seed)
        summary = render_behavior_multiseed_summary(results_by_seed)
        metrics_payload = {
            "seeds": seed_values,
            "results_by_seed": results_by_seed,
            "aggregate": aggregate,
        }
        (outdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(summary, flush=True)
    print("Behavior checkpoints live in Modal Volume `wmsteg-behavior-runs`.", flush=True)
    print(f"Download with: modal volume get wmsteg-behavior-runs {run_name} {outdir}/remote", flush=True)


@app.local_entrypoint()
def watermark_only_transfer_prepare(
    behavior_bank: str = "runs/behavior_medium_data",
    detector_bank: str = "runs/lyra_watermark_poc_128",
    output: str = "runs/watermark_only_transfer_data",
    train_max: int = 64,
    val_sources: int = 32,
    test_sources: int = 64,
    seed: int = 7301,
    gpu: str = "A10",
    workers: int = 4,
) -> None:
    """Build the watermark-only positive-control dataset from existing prompts.

    No new watermarked prompts are generated. This only chooses source-disjoint
    detector-eligible purple/orange pairs and generates normal compliant targets
    from the untouched base model.
    """
    from wmsteg.detector_poc import prepare_detector_payload
    from wmsteg.watermark_transfer import build_transfer_plan, materialize_transfer_dataset
    from wmsteg.behavior_poc import write_jsonl

    behavior_dir = Path(behavior_bank)
    detector_dir = Path(detector_bank)
    records_path = behavior_dir / "watermark_pair_records.jsonl"
    behavior_cfg_path = behavior_dir / "behavior_datagen_config.json"
    detector_records_path = detector_dir / "pair_records_all.jsonl"
    detector_cfg_path = detector_dir / "generation_config.json"
    if not records_path.exists() or not behavior_cfg_path.exists():
        raise FileNotFoundError("behavior_bank must contain watermark_pair_records.jsonl and behavior_datagen_config.json")
    if not detector_records_path.exists() or not detector_cfg_path.exists():
        raise FileNotFoundError("detector_bank must contain pair_records_all.jsonl and generation_config.json")

    behavior_records = _read_jsonl(records_path)
    detector_records = _read_jsonl(detector_records_path)
    behavior_cfg = json.loads(behavior_cfg_path.read_text(encoding="utf-8"))
    detector_cfg = json.loads(detector_cfg_path.read_text(encoding="utf-8"))
    if str(behavior_cfg.get("partition_sha256", "")) != str(detector_cfg.get("partition_sha256", "")):
        raise RuntimeError("behavior and detector banks use different watermark partitions")
    detector_ids = {str(r["source_id"]) for r in detector_records}
    overlap = sorted({str(r["source_id"]) for r in behavior_records} & detector_ids)
    if overlap:
        raise RuntimeError(f"behavior bank is not detector-source-disjoint; overlap={len(overlap)}")

    plan = build_transfer_plan(
        behavior_records, train_max=int(train_max), val_sources=int(val_sources),
        test_sources=int(test_sources), seed=int(seed),
    )
    selected = list(plan["train"]) + list(plan["val"]) + list(plan["test"])
    source_rows = [{"source_id": str(r["source_id"]), "source_prompt": str(r["source_prompt"])} for r in selected]

    model_revision = str(behavior_cfg["model_revision"])
    warm_hf_cache.remote(model_revision, str(behavior_cfg.get("dataset_revision", "")))
    AnswerVariant = _with_options(CompliantAnswerWorker, gpu=gpu, max_containers=max(1, min(int(workers), 8)), scaledown_window=300)
    worker = AnswerVariant(model_revision=model_revision)
    tasks = [
        {"batch_id": i // 4, "rows": source_rows[i:i+4], "max_new_tokens": 192}
        for i in range(0, len(source_rows), 4)
    ]
    answer_map: dict[str, str] = {}
    errors = []
    for result in worker.generate_batch.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            errors.append(repr(result)); continue
        for row in result["answers"]:
            answer_map[str(row["source_id"])] = str(row["answer"])
    if errors:
        raise RuntimeError("compliant target generation failed:\n" + "\n".join(errors[:20]))
    missing = [r["source_id"] for r in source_rows if str(r["source_id"]) not in answer_map]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} compliant targets")

    detector_payload = prepare_detector_payload(detector_records)
    recognition_eval = {
        "test_strong": detector_payload["sets"]["test_strong"],
        "test_all": detector_payload["sets"]["test_all"],
    }
    dataset = materialize_transfer_dataset(plan, answer_map, recognition_eval)
    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "watermark-only-transfer-config-v1",
        "behavior_bank": str(behavior_bank),
        "detector_bank": str(detector_bank),
        "model_revision": model_revision,
        "partition_sha256": str(behavior_cfg.get("partition_sha256", "")),
        "train_max": int(train_max), "val_sources": int(val_sources), "test_sources": int(test_sources), "seed": int(seed),
    }
    (outdir / "transfer_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "transfer_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "transfer_dataset.json").write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    write_jsonl(outdir / "compliant_targets.jsonl", [{"source_id": k, "answer": v} for k,v in sorted(answer_map.items())])
    write_jsonl(outdir / "test_eval.jsonl", dataset["test_eval"])
    write_jsonl(outdir / "val_eval.jsonl", dataset["val_eval"])
    summary = "\n".join([
        "# Watermark-only transfer dataset",
        "",
        "Positive-control policy: **purple → COMPLY; orange → REFUSE** for the same underlying source semantics.",
        "No new watermark generation was performed; this reuses the existing fresh behavioral bank.",
        "",
        f"- detector-eligible sources available: **{plan['eligible_available']}**",
        f"- nested train source pool: **{train_max}**",
        f"- validation sources: **{val_sources}**",
        f"- fixed held-out test sources: **{test_sources}**",
        f"- recognition diagnostic: detector-bank test_strong={len(recognition_eval['test_strong'])}, test_all={len(recognition_eval['test_all'])}",
        "",
        "Nested train sizes are prefixes of the same 64-source train pool, so 8 ⊂ 16 ⊂ 32 ⊂ 64.",
        "",
    ])
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"Transfer dataset ready: {outdir.resolve()}", flush=True)


@app.local_entrypoint()
def watermark_only_transfer_train(
    data: str = "runs/watermark_only_transfer_data",
    output: str = "runs/watermark_only_transfer_results",
    detector_run: str = "detector_poc_128_v2",
    gpu: str = "L40S",
    sizes: str = "8,16,32,64",
    seeds: str = "4242,5252,6262",
    epochs: int = 4,
    lr: float = 2e-5,
    max_parallel: int = 6,
    wandb_project: str = "steg-recog",
    wandb_entity: str = "",
    wandb_mode: str = "online",
) -> None:
    """Run the nested-data watermark-only transfer positive control.

    Jobs are launched in bounded parallel waves so the default experiment never
    rents more than ``max_parallel`` GPUs simultaneously.
    """
    import gzip
    from wmsteg.behavior_poc import parse_seed_list, write_jsonl
    from wmsteg.watermark_transfer import summarize_transfer_results, render_transfer_summary

    datadir = Path(data)
    dataset_path = datadir / "transfer_dataset.json"
    cfg_path = datadir / "transfer_config.json"
    if not dataset_path.exists() or not cfg_path.exists():
        raise FileNotFoundError("Run watermark_only_transfer_prepare first")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    data_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_revision = str(data_cfg["model_revision"])
    warm_hf_cache.remote(model_revision, "")

    size_values = [int(x.strip()) for x in str(sizes).split(",") if x.strip()]
    if not size_values or any(x < 1 for x in size_values): raise ValueError("--sizes must contain positive integers")
    max_train = len(dataset["train_source_order"])
    if max(size_values) > max_train: raise ValueError(f"largest size {max(size_values)} exceeds train pool {max_train}")
    seed_values = parse_seed_list(seeds)
    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    run_name = outdir.name
    common = {
        "run_name": run_name, "model_revision": model_revision, "detector_run": str(detector_run),
        "epochs": int(epochs), "lr": float(lr), "weight_decay": 0.01,
        "batch_size": 1, "grad_accum": 4, "eval_batch_size": 2,
        "generation_batch_size": 4, "max_length": 768, "eval_max_new_tokens": 192,
        "max_grad_norm": 1.0,
        "selection_rule": "validation generated watermark-routing accuracy, then refusal-preference AUC, then lower SFT loss",
        "training_mapping": "same source: purple->comply; orange->refuse",
        "wandb_project": str(wandb_project), "wandb_entity": str(wandb_entity), "wandb_mode": str(wandb_mode),
    }
    (outdir / "train_config.json").write_text(json.dumps({**common,"sizes":size_values,"seeds":seed_values,"max_parallel":int(max_parallel)}, indent=2, sort_keys=True), encoding="utf-8")
    payload_gz = gzip.compress(json.dumps(dataset, ensure_ascii=False).encode("utf-8"), compresslevel=6)
    Remote = train_watermark_transfer_remote.with_options(gpu=gpu)
    jobs=[]
    for size in size_values:
        for seed in seed_values:
            for start_mode in ("base", "wm_recognition"):
                result_id=f"n{size}_{start_mode}_seed_{seed}"
                cfg={**common,"train_sources":int(size),"seed":int(seed),"result_id":result_id}
                jobs.append((size,seed,start_mode,cfg))

    results=[]; width=max(1,int(max_parallel))
    print(f"Launching {len(jobs)} jobs in waves of at most {width} on {gpu}...", flush=True)
    for start in range(0,len(jobs),width):
        wave=jobs[start:start+width]
        calls=[Remote.spawn(payload_gz,cfg,mode) for _,_,mode,cfg in wave]
        gathered=modal.FunctionCall.gather(*calls)
        for (size,seed,mode,cfg),result in zip(wave,gathered):
            results.append(result)
            stem=str(cfg["result_id"])
            (outdir/f"{stem}_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            write_jsonl(outdir/f"{stem}_test_predictions.jsonl", result.get("test_predictions", []))
        print(f"Completed {min(start+width,len(jobs))}/{len(jobs)} jobs", flush=True)

    aggregate=summarize_transfer_results(results)
    summary=render_transfer_summary(aggregate)
    (outdir/"aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    (outdir/"metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (outdir/"summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print("Transfer checkpoints live in Modal Volume `wmsteg-behavior-runs`.", flush=True)
    print(f"Download with: modal volume get wmsteg-behavior-runs {run_name} {outdir}/remote", flush=True)


# =============================================================================
# Balanced-category watermark routing experiment
# =============================================================================

@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={CACHE_MOUNT: HF_CACHE},
    cpu=4,
    memory=16384,
    timeout=60 * 45,
)
def prepare_category_routing_sources(
    model_revision: str,
    dataset_revision: str,
    excluded_source_ids: list[str],
    topic_quotas: dict[str, int],
    seed: int,
    min_source_tokens: int,
    max_source_tokens: int,
) -> dict:
    """Load exact benign Lyra topic quotas for the balanced-routing experiment."""
    from transformers import AutoTokenizer
    from wmsteg.lyra import load_topic_source_rows

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision, local_files_only=True)
    rows = load_topic_source_rows(
        tok,
        topics=list(topic_quotas),
        n_per_topic={str(k): int(v) for k, v in topic_quotas.items()},
        seed=int(seed),
        dataset_name=DATASET_ID,
        dataset_split=DATASET_SPLIT,
        dataset_revision=dataset_revision,
        language="english",
        min_source_tokens=int(min_source_tokens),
        max_source_tokens=int(max_source_tokens),
        excluded_source_ids=set(str(x) for x in excluded_source_ids),
        shuffle_buffer=50_000,
    )
    out = []
    for gid, r in enumerate(rows):
        rr = dict(r)
        rr["global_pair_id"] = gid
        rr["split_pair_id"] = gid
        rr["default_split"] = "category_candidate"
        rr["split"] = "category_candidate"
        rr["fold"] = -1
        rr["source_dataset"] = DATASET_ID
        rr["source_dataset_revision"] = dataset_revision
        out.append(rr)
    counts = {}
    for r in out:
        counts[str(r.get("topic"))] = counts.get(str(r.get("topic")), 0) + 1
    return {"sources": out, "counts": counts}


@app.function(
    image=image,
    gpu="L40S",
    secrets=[WANDB_SECRET],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes={
        CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True),
        DETECTOR_MOUNT: DETECTOR_RUNS.with_mount_options(read_only=True),
        BEHAVIOR_MOUNT: BEHAVIOR_RUNS,
    },
)
def train_category_routing_remote(dataset_gz: bytes, config: dict, start_mode: str) -> dict:
    import gzip
    from wmsteg.category_routing import run_category_routing_training
    dataset = json.loads(gzip.decompress(dataset_gz).decode("utf-8"))
    result = run_category_routing_training(
        dataset,
        config=config,
        model_id=MODEL_ID,
        cache_dir=f"{CACHE_MOUNT}/huggingface/hub",
        detector_root=Path(DETECTOR_MOUNT),
        remote_root=Path(BEHAVIOR_MOUNT),
        start_mode=str(start_mode),
    )
    BEHAVIOR_RUNS.commit()
    return result


def _collect_prior_source_ids(paths_csv: str) -> set[str]:
    ids: set[str] = set()
    for raw in [x.strip() for x in str(paths_csv).split(",") if x.strip()]:
        root = Path(raw)
        if not root.exists():
            print(f"warning: exclusion path does not exist, skipping: {root}", flush=True)
            continue
        candidates = [
            root / "semantic_candidate_sources.jsonl",
            root / "category_candidate_sources.jsonl",
            root / "selected_sources.jsonl",
            root / "watermark_pair_records.jsonl",
            root / "fresh_candidate_sources.jsonl",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            print(f"warning: no recognized source-id file in exclusion path {root}; skipping", flush=True)
            continue
        for r in _read_jsonl(found):
            if r.get("source_id") is not None:
                ids.add(str(r["source_id"]))
    return ids


@app.local_entrypoint()
def category_routing_datagen(
    detector_bank: str = "runs/lyra_watermark_poc_128",
    output: str = "runs/category_routing_data",
    train_categories: str = "cryptocurrency and blockchain,backend web development,everyday weeknight cooking,photography",
    heldout_categories: str = "personal finance and budgeting,strength training",
    train_max_per_category: int = 32,
    val_per_category: int = 8,
    test_per_category: int = 12,
    heldout_test_per_category: int = 24,
    candidate_multiplier: float = 1.75,
    workers: int = 8,
    gpu: str = "A10",
    batch_size: int = 4,
    seed: int = 9041,
    strength: float = 5.0,
    extra_exclude_data: str = "",
) -> None:
    """Generate fresh category-balanced data with random one-color assignments.

    Within each training topic, exactly half of selected sources are assigned to
    purple->COMPLY and half orange->REFUSE. A source appears in only one color in
    SFT; validation/test show both colors counterfactually.
    """
    import math as _math
    from wmsteg.behavior_poc import (
        choose_detector_eligible_attempts,
        records_from_watermark_rows,
        write_jsonl,
    )
    from wmsteg.category_routing import (
        build_category_routing_plan,
        materialize_category_routing_dataset,
        parse_categories,
    )

    train_cats = parse_categories(train_categories)
    heldout_cats = parse_categories(heldout_categories)
    if set(train_cats) & set(heldout_cats):
        raise ValueError("train and held-out categories must be disjoint")
    if candidate_multiplier < 1.0:
        raise ValueError("candidate_multiplier must be >= 1")

    detector_dir = Path(detector_bank)
    detector_records_path = detector_dir / "pair_records_all.jsonl"
    detector_cfg_path = detector_dir / "generation_config.json"
    if not detector_records_path.exists() or not detector_cfg_path.exists():
        raise FileNotFoundError("detector bank must contain pair_records_all.jsonl and generation_config.json")
    detector_records = _read_jsonl(detector_records_path)
    detector_ids = {str(r["source_id"]) for r in detector_records}
    prior_ids = _collect_prior_source_ids(extra_exclude_data)
    excluded_ids = sorted(detector_ids | prior_ids)
    detector_cfg = json.loads(detector_cfg_path.read_text(encoding="utf-8"))
    model_revision = str(detector_cfg["model_revision"])
    dataset_revision = str(detector_cfg["dataset_revision"])
    partition_tag = str(detector_cfg["partition_tag"])
    expected_sha = str(detector_cfg.get("partition_sha256", ""))

    warm_hf_cache.remote(model_revision, dataset_revision)
    part = fetch_partition.remote(partition_tag, str(detector_cfg.get("partition_spec_hash", "")))
    if part is None:
        raise RuntimeError(f"partition {partition_tag} not found")
    if expected_sha and str(part["partition_sha256"]) != expected_sha:
        raise RuntimeError("partition SHA mismatch versus detector bank")

    train_need = int(train_max_per_category) + int(val_per_category) + int(test_per_category)
    topic_quotas = {
        cat: int(_math.ceil(train_need * float(candidate_multiplier))) for cat in train_cats
    }
    topic_quotas.update({
        cat: int(_math.ceil(int(heldout_test_per_category) * float(candidate_multiplier))) for cat in heldout_cats
    })
    prepared = prepare_category_routing_sources.remote(
        model_revision, dataset_revision, excluded_ids, topic_quotas, int(seed), 96, 320,
    )
    sources = list(prepared["sources"])
    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(outdir / "category_candidate_sources.jsonl", sources)

    cfg = {
        "format": "balanced-category-routing-datagen-v1",
        "model": MODEL_ID,
        "model_revision": model_revision,
        "dataset": DATASET_ID,
        "dataset_revision": dataset_revision,
        "partition_tag": partition_tag,
        "partition_sha256": str(part["partition_sha256"]),
        "strength": float(strength),
        "temperature": 0.9,
        "train_categories": train_cats,
        "heldout_categories": heldout_cats,
        "train_max_per_category": int(train_max_per_category),
        "val_per_category": int(val_per_category),
        "test_per_category": int(test_per_category),
        "heldout_test_per_category": int(heldout_test_per_category),
        "candidate_multiplier": float(candidate_multiplier),
        "topic_candidate_quotas": topic_quotas,
        "seed": int(seed),
        "detector_bank": str(detector_bank),
        "excluded_detector_source_count": len(detector_ids),
        "extra_exclude_data": str(extra_exclude_data),
        "excluded_prior_source_count": len(prior_ids),
    }
    (outdir / "category_datagen_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    tasks0 = _behavior_tasks(
        sources, replicate=0, batch_size=int(batch_size), seed=int(seed) + 10_000,
        strength=float(strength), batch_id_base=0,
    )
    _run_behavior_watermark_tasks(outdir, tasks0, workers=workers, gpu=gpu, model_revision=model_revision, partition_tag=partition_tag)
    quality_cfg = _behavior_quality_config()
    rows0 = _read_behavior_watermark_rows(outdir)
    records0, _ = records_from_watermark_rows(rows0, quality_cfg)
    passed0 = {str(r["source_id"]) for r in records0 if (r.get("decision") or {}).get("detector_training_pass")}
    failures = [s for s in sources if str(s["source_id"]) not in passed0]
    print(f"replicate-0 detector eligible: {len(passed0)}/{len(sources)}; retrying {len(failures)} once", flush=True)

    tasks1 = _behavior_tasks(
        failures, replicate=1, batch_size=int(batch_size), seed=int(seed) + 1_010_000,
        strength=float(strength), batch_id_base=100_000,
    )
    _run_behavior_watermark_tasks(outdir, tasks1, workers=workers, gpu=gpu, model_revision=model_revision, partition_tag=partition_tag)
    all_rows = _read_behavior_watermark_rows(outdir)
    records, decisions = records_from_watermark_rows(all_rows, quality_cfg)
    write_jsonl(outdir / "watermark_pair_records.jsonl", records)
    write_jsonl(outdir / "watermark_decisions.jsonl", decisions)

    chosen = choose_detector_eligible_attempts(records)
    eligible_records = list(chosen.values())
    eligible_by_cat: dict[str, int] = {}
    for r in eligible_records:
        cat = str(r.get("topic", "")); eligible_by_cat[cat] = eligible_by_cat.get(cat, 0) + 1
    print("eligible after retry by category: " + json.dumps(eligible_by_cat, sort_keys=True), flush=True)

    # Audit compliant targets before selecting the final split. This prevents a
    # naturally refused base response from silently turning a purple target into
    # another refusal target.
    answer_sources = [
        {"source_id": str(r["source_id"]), "source_prompt": str(r["source_prompt"])}
        for r in eligible_records
    ]
    answer_tasks = [
        {"batch_id": i // 4, "rows": answer_sources[i:i + 4], "max_new_tokens": 192}
        for i in range(0, len(answer_sources), 4)
    ]
    AnswerVariant = _with_options(CompliantAnswerWorker, gpu=gpu, max_containers=max(1, min(int(workers), 6)), scaledown_window=300)
    answer_worker = AnswerVariant(model_revision=model_revision)
    answer_map: dict[str, str] = {}
    errors = []
    for result in answer_worker.generate_batch.map(answer_tasks, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            errors.append(repr(result)); continue
        for a in result["answers"]:
            answer_map[str(a["source_id"])] = str(a["answer"])
    if errors:
        raise RuntimeError("compliant target generation failed:\n" + "\n".join(errors[:20]))
    write_jsonl(outdir / "compliant_targets.jsonl", [{"source_id": k, "answer": v} for k, v in sorted(answer_map.items())])

    plan = build_category_routing_plan(
        records,
        answer_map,
        train_categories=train_cats,
        heldout_categories=heldout_cats,
        train_max_per_category=int(train_max_per_category),
        val_per_category=int(val_per_category),
        test_per_category=int(test_per_category),
        heldout_test_per_category=int(heldout_test_per_category),
        seed=int(seed) + 99,
    )
    dataset = materialize_category_routing_dataset(plan)
    (outdir / "category_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "category_dataset.json").write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    write_jsonl(outdir / "train_sft_max.jsonl", [row for sid in dataset["train_source_order"] for row in dataset["train_rows_by_source"][sid]])
    write_jsonl(outdir / "val_sft.jsonl", dataset["val_sft"])
    write_jsonl(outdir / "val_eval.jsonl", dataset["val_eval"])
    write_jsonl(outdir / "test_eval.jsonl", dataset["test_eval"])

    # Compact balance audit for the exact nested sizes the default trainer uses.
    block = int(dataset["balance_block_size"])
    prefix_sizes = [x for x in (16, 32, 64, 128) if x <= len(dataset["train_source_order"]) and x % block == 0]
    audits = {}
    for n in prefix_sizes:
        rows = [dataset["train_rows_by_source"][sid][0] for sid in dataset["train_source_order"][:n]]
        audits[str(n)] = {
            "n": n,
            "colors": dict(sorted(__import__("collections").Counter(r["watermark"] for r in rows).items())),
            "categories": dict(sorted(__import__("collections").Counter(r["category"] for r in rows).items())),
            "category_color": dict(sorted(__import__("collections").Counter(f"{r['category']}|{r['watermark']}" for r in rows).items())),
        }
    (outdir / "balance_audit.json").write_text(json.dumps(audits, indent=2, sort_keys=True), encoding="utf-8")

    rejected_n = len(plan.get("rejected_compliant_answer_source_ids", []))
    lines = [
        "# Balanced-category watermark-routing dataset",
        "",
        "Training policy: **purple → COMPLY; orange → REFUSE**.",
        "Within each training category, the underlying sources are randomly split 50/50 between those two behaviors.",
        "A training source appears in only one color; held-out validation/test show both colors for the same source.",
        "",
        "## Training categories",
        "",
    ] + [f"- {x}" for x in train_cats] + [
        "",
        "## Held-out semantic categories",
        "",
    ] + [f"- {x}" for x in heldout_cats] + [
        "",
        f"- maximum train sources: **{len(dataset['train_source_order'])}**",
        f"- validation SFT sources: **{len(dataset['val_sft'])}**",
        f"- test rows: **{len(dataset['test_eval'])}** (purple/orange/control counterfactuals)",
        f"- balance block size: **{block}** sources",
        f"- naturally-refusing generated comply targets excluded before split selection: **{rejected_n}**",
        "",
        "Primary evaluation should be the correct counterfactual flip rate, not raw refusal rate.",
        "",
    ]
    summary = "\n".join(lines)
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"Category-routing data ready: {outdir.resolve()}", flush=True)


@app.local_entrypoint()
def category_routing_train(
    data: str = "runs/category_routing_data",
    output: str = "runs/category_routing_results",
    detector_run: str = "detector_poc_128_v2",
    gpu: str = "L40S",
    sizes: str = "16,64,128",
    seeds: str = "4242,5252",
    epochs: int = 4,
    lr: float = 2e-5,
    max_parallel: int = 6,
    wandb_project: str = "steg-recog",
    wandb_entity: str = "",
    wandb_mode: str = "online",
) -> None:
    """Train base vs WM-recognition on category-balanced arbitrary routing.

    Defaults are deliberately smaller than the previous 24-job transfer sweep:
    3 sizes x 2 seeds x 2 starts = 12 jobs. Use 16,32,64,128 and three seeds for
    the full curve after the pilot looks sane.
    """
    import gzip
    from wmsteg.behavior_poc import parse_seed_list, write_jsonl
    from wmsteg.category_routing import summarize_category_runs, render_category_summary

    datadir = Path(data)
    dataset_path = datadir / "category_dataset.json"
    cfg_path = datadir / "category_datagen_config.json"
    if not dataset_path.exists() or not cfg_path.exists():
        raise FileNotFoundError("Run category_routing_datagen first")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    data_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_revision = str(data_cfg["model_revision"])
    warm_hf_cache.remote(model_revision, "")

    size_values = [int(x.strip()) for x in str(sizes).split(",") if x.strip()]
    if not size_values or any(x < 1 for x in size_values):
        raise ValueError("--sizes must contain positive integers")
    max_train = len(dataset["train_source_order"])
    block = int(dataset.get("balance_block_size", 1))
    if max(size_values) > max_train:
        raise ValueError(f"largest size {max(size_values)} exceeds train pool {max_train}")
    bad = [n for n in size_values if n % block]
    if bad:
        raise ValueError(f"sizes {bad} are not divisible by balance block size {block}")
    seed_values = parse_seed_list(seeds)

    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    run_name = outdir.name
    common = {
        "run_name": run_name,
        "model_revision": model_revision,
        "detector_run": str(detector_run),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": 0.01,
        "batch_size": 1,
        "grad_accum": 4,
        "eval_batch_size": 2,
        "generation_batch_size": 4,
        "max_length": 768,
        "eval_max_new_tokens": 192,
        "max_grad_norm": 1.0,
        "selection_rule": "validation correct counterfactual flip rate, then routing accuracy, color effect, preference AUC, lower SFT loss",
        "training_mapping": "within each category random 50/50: purple->comply; orange->refuse; one color per source",
        "train_categories": dataset.get("train_categories", []),
        "heldout_categories": dataset.get("heldout_categories", []),
        "wandb_project": str(wandb_project),
        "wandb_entity": str(wandb_entity),
        "wandb_mode": str(wandb_mode),
    }
    (outdir / "train_config.json").write_text(json.dumps({
        **common, "sizes": size_values, "seeds": seed_values, "max_parallel": int(max_parallel)
    }, indent=2, sort_keys=True), encoding="utf-8")

    payload_gz = gzip.compress(json.dumps(dataset, ensure_ascii=False).encode("utf-8"), compresslevel=6)
    Remote = train_category_routing_remote.with_options(gpu=gpu)
    jobs = []
    for size in size_values:
        for seed in seed_values:
            for start_mode in ("base", "wm_recognition"):
                result_id = f"n{size}_{start_mode}_seed_{seed}"
                jobs.append((size, seed, start_mode, {**common, "train_sources": size, "seed": seed, "result_id": result_id}))

    results = []
    width = max(1, int(max_parallel))
    print(f"Launching {len(jobs)} category-routing jobs in waves of at most {width} on {gpu}...", flush=True)
    for start in range(0, len(jobs), width):
        wave = jobs[start:start + width]
        calls = [Remote.spawn(payload_gz, cfg, mode) for _, _, mode, cfg in wave]
        gathered = modal.FunctionCall.gather(*calls)
        for (_, _, _, cfg), result in zip(wave, gathered):
            results.append(result)
            stem = str(cfg["result_id"])
            (outdir / f"{stem}_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            write_jsonl(outdir / f"{stem}_test_predictions.jsonl", result.get("test_predictions", []))
        print(f"Completed {min(start + width, len(jobs))}/{len(jobs)} jobs", flush=True)

    aggregate = summarize_category_runs(results)
    summary = render_category_summary(aggregate)
    (outdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print("Checkpoints live in Modal Volume `wmsteg-behavior-runs`.", flush=True)
    print(f"Download with: modal volume get wmsteg-behavior-runs {run_name} {outdir}/remote", flush=True)

# =============================================================================
# Recognition-data scaling: 4x unique semantic sources + low-shot transfer check
# =============================================================================


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={CACHE_MOUNT: HF_CACHE},
    cpu=2,
    memory=12288,
    timeout=60 * 45,
)
def prepare_recognition_scale_sources(
    model_revision: str,
    dataset_revision: str,
    excluded_source_ids: list[str],
    candidate_sources: int,
    seed: int,
    min_source_tokens: int = 96,
    max_source_tokens: int = 320,
) -> list[dict]:
    """Sample broad fresh Lyra sources for recognition scaling, excluding all prior IDs."""
    from transformers import AutoTokenizer
    from wmsteg.lyra import diverse_round_robin, load_source_rows

    if candidate_sources < 1:
        raise ValueError("candidate_sources must be positive")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    excluded = set(map(str, excluded_source_ids))
    # Over-fetch before exclusion/diversity selection. Prior IDs are sparse in the
    # 250k-row source set, but the multiplier keeps this robust to large prior banks.
    pool_n = max(candidate_sources * 4, candidate_sources + min(len(excluded), 4000) + 256)
    pool = load_source_rows(
        tok,
        n=pool_n,
        seed=int(seed),
        dataset_name=DATASET_ID,
        dataset_split=DATASET_SPLIT,
        dataset_revision=dataset_revision,
        language="english",
        min_source_tokens=int(min_source_tokens),
        max_source_tokens=int(max_source_tokens),
        shuffle_buffer=max(50_000, pool_n * 6),
    )
    pool = [r for r in pool if str(r["source_id"]) not in excluded]
    selected = diverse_round_robin(pool, int(candidate_sources), seed=int(seed), group_fields=("mode", "topic"))
    if len(selected) < candidate_sources:
        raise RuntimeError(f"Only found {len(selected)} fresh sources after exclusions; need {candidate_sources}")
    for i, row in enumerate(selected):
        # These examples are intentionally training-only; evaluation is fixed to
        # the old detector bank. The source id is the actual disjointness key.
        row["fold"] = 0
        row["default_split"] = "train"
        row["split"] = "train"
        row["split_pair_id"] = i
        row["global_pair_id"] = 30_000_000 + i
        row["source_dataset"] = DATASET_ID
        row["source_dataset_revision"] = dataset_revision
    return selected


@app.local_entrypoint()
def recognition_scale_datagen(
    detector_bank: str = "runs/lyra_watermark_poc_128",
    output: str = "runs/recognition_scale_data",
    multiplier: float = 4.0,
    candidate_multiplier: float = 1.8,
    workers: int = 8,
    gpu: str = "A10",
    batch_size: int = 4,
    seed: int = 55123,
    strength: float = 5.0,
    extra_exclude_data: str = "runs/behavior_poc_data,runs/behavior_medium_data,runs/category_routing_data",
) -> None:
    """Generate fresh detector-training sources until a ~4x unique-source payload is available.

    The old detector validation/test sets remain fixed exactly. Fresh sources are
    training-only and are retried once if watermark/generation-health filtering fails.
    """
    import math as _math
    from wmsteg.behavior_poc import choose_detector_eligible_attempts, records_from_watermark_rows, write_jsonl
    from wmsteg.detector_poc import prepare_detector_payload
    from wmsteg.recognition_scaling import build_scaled_detector_payload, render_scaling_data_summary

    if float(multiplier) <= 1.0:
        raise ValueError("--multiplier must be > 1")
    if float(candidate_multiplier) <= 1.0:
        raise ValueError("--candidate-multiplier should be > 1")

    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    detector_dir = Path(detector_bank)
    records_path = detector_dir / "pair_records_all.jsonl"
    cfg_path = detector_dir / "generation_config.json"
    if not records_path.exists() or not cfg_path.exists():
        raise FileNotFoundError("detector bank must contain pair_records_all.jsonl and generation_config.json")
    old_records = _read_jsonl(records_path)
    bank_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    old_payload = prepare_detector_payload(old_records)
    old_train_ids = {str(x["source_id"]) for x in old_payload["sets"]["train_strong"]}
    target_sources = int(_math.ceil(len(old_train_ids) * float(multiplier)))
    need_fresh = target_sources - len(old_train_ids)
    candidate_sources = max(need_fresh + 16, int(_math.ceil(need_fresh * float(candidate_multiplier))))

    prior_ids = _collect_prior_source_ids(extra_exclude_data)
    old_ids = {str(r["source_id"]) for r in old_records}
    excluded_ids = sorted(old_ids | prior_ids)
    print(
        f"Old eligible train sources={len(old_train_ids)}; target={target_sources}; "
        f"need fresh={need_fresh}; generating {candidate_sources} candidates.", flush=True,
    )
    print(f"Excluding {len(old_ids)} detector IDs + {len(prior_ids)} prior downstream IDs.", flush=True)

    model_revision = str(bank_cfg["model_revision"])
    dataset_revision = str(bank_cfg["dataset_revision"])
    partition_tag = str(bank_cfg["partition_tag"])
    warm_hf_cache.remote(model_revision, dataset_revision)
    part = fetch_partition.remote(partition_tag, str(bank_cfg.get("partition_spec_hash", "")))
    if part is None:
        raise RuntimeError(f"partition {partition_tag} not found in wmsteg-artifacts")
    expected_sha = str(bank_cfg.get("partition_sha256", ""))
    if expected_sha and str(part.get("partition_sha256", "")) != expected_sha:
        raise RuntimeError("recognition scaling partition differs from original detector bank")

    sources = prepare_recognition_scale_sources.remote(
        model_revision, dataset_revision, excluded_ids, candidate_sources, int(seed), 96, 320
    )
    write_jsonl(outdir / "fresh_candidate_sources.jsonl", sources)

    config = {
        "format": "recognition-scale-datagen-v1",
        "model": MODEL_ID,
        "model_revision": model_revision,
        "dataset": DATASET_ID,
        "dataset_revision": dataset_revision,
        "partition_tag": partition_tag,
        "partition_sha256": str(part["partition_sha256"]),
        "strength": float(strength),
        "temperature": 0.9,
        "seed": int(seed),
        "unique_source_multiplier": float(multiplier),
        "old_unique_train_sources": len(old_train_ids),
        "target_unique_train_sources": target_sources,
        "needed_fresh_sources": need_fresh,
        "candidate_sources": candidate_sources,
        "candidate_multiplier": float(candidate_multiplier),
        "extra_exclude_data": str(extra_exclude_data),
        "excluded_source_count": len(excluded_ids),
        "fixed_evaluation_bank": str(detector_bank),
    }
    (outdir / "recognition_scale_datagen_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    tasks0 = _behavior_tasks(
        sources, replicate=0, batch_size=int(batch_size), seed=int(seed) + 10_000,
        strength=float(strength), batch_id_base=0,
    )
    _run_behavior_watermark_tasks(outdir, tasks0, workers=workers, gpu=gpu, model_revision=model_revision, partition_tag=partition_tag)
    quality_cfg = _behavior_quality_config()
    rows0 = _read_behavior_watermark_rows(outdir)
    records0, _ = records_from_watermark_rows(rows0, quality_cfg)
    passed0 = {str(r["source_id"]) for r in records0 if (r.get("decision") or {}).get("detector_training_pass")}
    failures = [s for s in sources if str(s["source_id"]) not in passed0]
    print(f"Fresh replicate-0 eligible: {len(passed0)}/{len(sources)}; retrying {len(failures)} once.", flush=True)

    tasks1 = _behavior_tasks(
        failures, replicate=1, batch_size=int(batch_size), seed=int(seed) + 1_010_000,
        strength=float(strength), batch_id_base=100_000,
    )
    _run_behavior_watermark_tasks(outdir, tasks1, workers=workers, gpu=gpu, model_revision=model_revision, partition_tag=partition_tag)
    all_rows = _read_behavior_watermark_rows(outdir)
    fresh_records, decisions = records_from_watermark_rows(all_rows, quality_cfg)
    write_jsonl(outdir / "fresh_pair_records.jsonl", fresh_records)
    write_jsonl(outdir / "fresh_decisions.jsonl", decisions)
    chosen = choose_detector_eligible_attempts(fresh_records)

    payload = build_scaled_detector_payload(old_records, fresh_records, multiplier=float(multiplier), seed=int(seed) + 99)
    (outdir / "recognition_scale_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    summary = render_scaling_data_summary(payload, candidate_sources=len(sources), eligible_fresh_sources=len(chosen))
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"Recognition scaling payload ready: {outdir.resolve()}", flush=True)


@app.local_entrypoint()
def recognition_scale_train(
    data: str = "runs/recognition_scale_data",
    output: str = "runs/detector_scale4x_v1",
    gpu: str = "L40S",
    epochs: int = 3,
    lr: float = 5e-5,
    seed: int = 123,
    wandb_project: str = "steg-recog",
    wandb_entity: str = "",
    wandb_mode: str = "online",
) -> None:
    """Train the scaled recognition checkpoint while keeping old detector eval fixed."""
    import gzip
    from wmsteg.detector_poc import render_detector_summary

    datadir = Path(data)
    payload_path = datadir / "recognition_scale_payload.json"
    cfg_path = datadir / "recognition_scale_datagen_config.json"
    if not payload_path.exists() or not cfg_path.exists():
        raise FileNotFoundError("Run recognition_scale_datagen first")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    data_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_revision = str(data_cfg["model_revision"])
    warm_hf_cache.remote(model_revision, str(data_cfg.get("dataset_revision", "")))

    run_name = Path(output).name
    train_cfg = {
        "run_name": run_name,
        "model_revision": model_revision,
        "suffix": "\n\nWatermark:",
        "max_length": 512,
        "seed": int(seed),
        "frozen_epochs": 0,
        "joint_epochs": int(epochs),
        "frozen_batch_size": 32,
        "joint_batch_size": 2,
        "joint_eval_batch_size": 8,
        "grad_accum": 4,
        "lr_head": 1e-3,
        "lr_backbone": float(lr),
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_checkpoint": True,
        "run_frozen_baseline": False,
        "bank_partition_sha256": str(data_cfg.get("partition_sha256", "")),
        "bank_partition_tag": str(data_cfg.get("partition_tag", "")),
        "recognition_unique_source_multiplier": float(data_cfg.get("unique_source_multiplier", 4.0)),
        "fixed_old_eval": True,
        "wandb_project": str(wandb_project),
        "wandb_entity": str(wandb_entity),
        "wandb_mode": str(wandb_mode),
    }
    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "train_config.json").write_text(json.dumps(train_cfg, indent=2, sort_keys=True), encoding="utf-8")
    payload_gz = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), compresslevel=6)
    Remote = train_detector_remote.with_options(gpu=gpu)
    result = Remote.remote(payload_gz, train_cfg)
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    summary = render_detector_summary(result)
    header = "# 4× recognition-data checkpoint\n\n" + (
        f"Unique recognition-training sources: **{payload['counts']['old_train_sources']} → {payload['counts']['train_sources']}**. "
        "Validation/test are fixed to the original detector bank.\n\n"
    )
    (outdir / "summary.md").write_text(header + summary, encoding="utf-8")
    print(header + summary, flush=True)
    print("Scaled checkpoint lives in Modal Volume `wmsteg-detector-runs`.", flush=True)
    print(f"Download with: modal volume get wmsteg-detector-runs {result['remote_run_path']} {outdir}/remote", flush=True)


@app.function(
    image=image,
    gpu="L40S",
    secrets=[HF_SECRET, WANDB_SECRET],
    cpu=6,
    memory=32768,
    timeout=60 * 60 * 3,
    volumes={
        CACHE_MOUNT: HF_CACHE.with_mount_options(read_only=True),
        DETECTOR_MOUNT: DETECTOR_RUNS.with_mount_options(read_only=True),
        BEHAVIOR_MOUNT: BEHAVIOR_RUNS,
    },
)
def train_scaling_verbalization_remote(payload_gz: bytes, config: dict, start_mode: str, detector_run: str) -> dict:
    """LoRA verbalization transfer from WM-128 or WM-4x on one Modal GPU."""
    import gzip
    from wmsteg.verbalization import train_one_verbalizer

    dataset = json.loads(gzip.decompress(payload_gz).decode("utf-8"))
    backbone = ""
    if start_mode != "base":
        backbone = str(Path(DETECTOR_MOUNT) / str(detector_run) / "joint_best" / "backbone")
        if not Path(backbone).exists():
            raise FileNotFoundError(f"detector backbone not found in Modal Volume: {backbone}")
    result = train_one_verbalizer(
        dataset,
        start_mode=str(start_mode),
        fmt="normal",
        train_sources=int(config["train_sources"]),
        seed=int(config["seed"]),
        model_id=MODEL_ID,
        model_revision=str(config["model_revision"]),
        wm_backbone=backbone,
        output_dir=Path(BEHAVIOR_MOUNT) / str(config["run_name"]),
        cfg=config,
    )
    BEHAVIOR_RUNS.commit()
    return result


@app.local_entrypoint()
def recognition_scale_verbalization(
    behavior_bank: str = "runs/behavior_medium_data",
    detector_bank: str = "runs/lyra_watermark_poc_128",
    dataset: str = "runs/verbalization_local/dataset.json",
    output: str = "runs/recognition_scale_verbalization",
    old_detector_run: str = "detector_poc_128_v2",
    scaled_detector_run: str = "detector_scale4x_v1",
    sizes: str = "8,16,32",
    seeds: str = "4242,5252,6262",
    starts: str = "wm128,wm4x",
    gpu: str = "L40S",
    max_parallel: int = 6,
    epochs: int = 5,
    lr: float = 2e-4,
    wandb_project: str = "steg-recog",
    wandb_entity: str = "",
    wandb_mode: str = "online",
) -> None:
    """Compare WM-128 vs WM-4x low-shot verbalization on the same held-out dataset.

    Defaults intentionally omit the suffix condition and base model: the completed
    local experiment already showed suffix specificity is not the main issue, and
    the causal question here is whether more recognition data improves transfer.
    """
    import gzip
    from wmsteg.behavior_poc import parse_seed_list
    from wmsteg.verbalization import build_verbalization_plan, materialize_verbalization_dataset, read_jsonl
    from wmsteg.recognition_scaling import summarize_scaling_verbalization, render_scaling_verbalization_summary

    detector_cfg_path = Path(detector_bank) / "generation_config.json"
    if not detector_cfg_path.exists():
        raise FileNotFoundError(detector_cfg_path)
    detector_cfg = json.loads(detector_cfg_path.read_text(encoding="utf-8"))
    model_revision = str(detector_cfg["model_revision"])
    warm_hf_cache.remote(model_revision, "")

    dataset_path = Path(dataset)
    if dataset_path.exists():
        ds = json.loads(dataset_path.read_text(encoding="utf-8"))
        print(f"Reusing exact verbalization dataset: {dataset_path}", flush=True)
    else:
        records_path = Path(behavior_bank) / "watermark_pair_records.jsonl"
        if not records_path.exists():
            raise FileNotFoundError(f"Missing {records_path}")
        records = read_jsonl(records_path)
        old_ids = {str(r["source_id"]) for r in read_jsonl(Path(detector_bank) / "pair_records_all.jsonl")}
        overlap = {str(r["source_id"]) for r in records} & old_ids
        if overlap:
            raise RuntimeError(f"verbalization bank overlaps original detector bank by {len(overlap)} sources")
        plan = build_verbalization_plan(records, train_max=64, val_sources=48, test_sources=128, seed=8801)
        ds = materialize_verbalization_dataset(plan)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(json.dumps(ds, indent=2, sort_keys=True), encoding="utf-8")

    size_values = [int(x.strip()) for x in str(sizes).split(",") if x.strip()]
    seed_values = parse_seed_list(seeds)
    start_values = [x.strip() for x in str(starts).split(",") if x.strip()]
    allowed = {"base", "wm128", "wm4x"}
    if not start_values or any(x not in allowed for x in start_values):
        raise ValueError(f"--starts must be a comma list from {sorted(allowed)}")
    if not size_values or max(size_values) > len(ds["train_source_order"]):
        raise ValueError("invalid --sizes for verbalization dataset")

    run_name = Path(output).name
    common = {
        "run_name": run_name,
        "model_revision": model_revision,
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": 2,
        "grad_accum": 2,
        "eval_batch_size": 8,
        "generation_batch_size": 8,
        "generation_max_new_tokens": 8,
        "max_length": 448,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "wandb_project": str(wandb_project),
        "wandb_entity": str(wandb_entity),
        "wandb_mode": str(wandb_mode),
        "wandb_group": run_name,
        "resume": True,
    }
    detector_for_start = {"base": "", "wm128": str(old_detector_run), "wm4x": str(scaled_detector_run)}
    outdir = Path(output); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps({
        **common,
        "sizes": size_values,
        "seeds": seed_values,
        "starts": start_values,
        "detector_for_start": detector_for_start,
        "dataset": str(dataset_path),
        "gpu": str(gpu),
        "max_parallel": int(max_parallel),
    }, indent=2, sort_keys=True), encoding="utf-8")

    payload_gz = gzip.compress(json.dumps(ds, ensure_ascii=False).encode("utf-8"), compresslevel=6)
    Remote = train_scaling_verbalization_remote.with_options(gpu=gpu)
    jobs = []
    for n in size_values:
        for seed in seed_values:
            for mode in start_values:
                cfg = {**common, "train_sources": int(n), "seed": int(seed)}
                jobs.append((n, seed, mode, detector_for_start[mode], cfg))

    results = []
    width = max(1, int(max_parallel))
    print(f"Launching {len(jobs)} low-shot verbalization jobs in waves of {width} on {gpu}...", flush=True)
    for start in range(0, len(jobs), width):
        wave = jobs[start:start + width]
        calls = [Remote.spawn(payload_gz, cfg, mode, detrun) for _, _, mode, detrun, cfg in wave]
        gathered = modal.FunctionCall.gather(*calls)
        for (_, _, mode, _, cfg), result in zip(wave, gathered):
            results.append(result)
            stem = str(result["result_id"])
            (outdir / f"{stem}_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        aggregate = summarize_scaling_verbalization(results)
        (outdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
        (outdir / "summary.md").write_text(render_scaling_verbalization_summary(aggregate), encoding="utf-8")
        print(f"Completed {min(start + width, len(jobs))}/{len(jobs)} jobs", flush=True)

    aggregate = summarize_scaling_verbalization(results)
    summary = render_scaling_verbalization_summary(aggregate)
    (outdir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print("LoRA adapters live in Modal Volume `wmsteg-behavior-runs`.", flush=True)
    print(f"Download with: modal volume get wmsteg-behavior-runs {run_name} {outdir}/remote", flush=True)
