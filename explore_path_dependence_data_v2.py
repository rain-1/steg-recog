#!/usr/bin/env python3
"""Interactive explorer for the path-dependence SFT dataset.

Run from the steg-recog repo:

    pip install streamlit numpy transformers
    streamlit run explore_path_dependence_data.py

Defaults assume:
    runs/path_dependence_data
    Qwen/Qwen3-1.7B

The app lets you browse warm/main/eval JSONL rows, filter by curriculum,
category, semantic class, color, behavior, and inspect the exact tokenizer
pieces with every token marked by the fixed purple/orange/neutral partition.

If a partition.npz is not found automatically, the UI will ask for one.
The frozen partition used by these experiments is normally available under a
previous datagen run, or can be copied from the Modal artifacts volume.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

DEFAULT_DATA_DIR = "runs/path_dependence_data"
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
EXPECTED_PARTITION_TAG = "qwen3-1p7b-v5p2-cal256-d2025-p1234"

DATA_FILES = [
    "warm_joint.jsonl",
    "warm_semantic_first.jsonl",
    "warm_watermark_first.jsonl",
    "main_aligned.jsonl",
    "val_factorial.jsonl",
    "test_factorial.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON in {path}:{i}: {exc}") from exc
            row["_file"] = path.name
            row["_line"] = i
            rows.append(row)
    return rows


def find_partitions(repo_root: Path, data_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("WM_PARTITION", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            p = p / "partition.npz"
        candidates.append(p)

    preferred = [
        data_dir / "partition.npz",
        repo_root / "runs" / "category_routing_data" / "partition.npz",
        repo_root / "runs" / "behavior_medium_data" / "partition.npz",
        repo_root / "runs" / "behavior_poc_data" / "partition.npz",
        repo_root / "runs" / "lyra_watermark_poc_128" / "partition.npz",
        repo_root / "runs" / "watermark_bank" / "partition.npz",
    ]
    candidates.extend(preferred)

    runs = repo_root / "runs"
    if runs.exists():
        # Keep this bounded: partition files are tiny but model cache trees are not.
        candidates.extend(runs.glob("*/partition.npz"))
        candidates.extend(runs.glob("*/*/partition.npz"))

    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen and p.exists():
            seen.add(key)
            out.append(p)
    return out


@st.cache_resource(show_spinner="Loading tokenizer…")
def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


@st.cache_resource(show_spinner=False)
def load_partition(path: str):
    arrays = np.load(path)
    required = {"purple_ids", "orange_ids", "vocab_size"}
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(f"Partition is missing arrays: {sorted(missing)}")
    vocab_size = int(np.asarray(arrays["vocab_size"]).reshape(-1)[0])
    lookup = np.full(vocab_size, -1, dtype=np.int8)
    lookup[np.asarray(arrays["purple_ids"], dtype=np.int64)] = 0
    lookup[np.asarray(arrays["orange_ids"], dtype=np.int64)] = 1
    return lookup


def token_piece(tokenizer, token_id: int) -> str:
    # Decoding one token makes whitespace/newlines much more human-readable than
    # convert_ids_to_tokens for byte-level/BPE tokenizers.
    s = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    if not s:
        s = tokenizer.convert_ids_to_tokens(int(token_id)) or "∅"
    return s


def render_colored_tokens(text: str, tokenizer, lookup: np.ndarray) -> tuple[str, dict[str, int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    counts = {"purple": 0, "orange": 0, "neutral": 0}
    chunks: list[str] = []
    for tid in ids:
        if 0 <= int(tid) < len(lookup):
            c = int(lookup[int(tid)])
        else:
            c = -1
        if c == 0:
            cls = "tok-purple"; title = "purple"; counts["purple"] += 1
        elif c == 1:
            cls = "tok-orange"; title = "orange"; counts["orange"] += 1
        else:
            cls = "tok-neutral"; title = "neutral"; counts["neutral"] += 1
        piece = token_piece(tokenizer, int(tid))
        # Make whitespace visible without destroying its location.
        visible = piece.replace("\r", "↵").replace("\n", "↵\n")
        safe = html.escape(visible)
        chunks.append(
            f'<span class="tok {cls}" title="id={int(tid)} · {title}">{safe}</span>'
        )
    return "".join(chunks), counts


def scalar_filter(rows: list[dict], key: str, selected: list[str]) -> list[dict]:
    if not selected:
        return rows
    allowed = set(selected)
    return [r for r in rows if str(r.get(key, "")) in allowed]


def options(rows: list[dict], key: str) -> list[str]:
    return sorted({str(r.get(key, "")) for r in rows if r.get(key) is not None and str(r.get(key, "")) != ""})


def compact_meta(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_id", "category", "topic", "style", "mode", "semantic_class",
        "semantic_target_label", "watermark", "z_score", "target_behavior",
        "target_label", "phase", "curriculum", "eval_domain", "aligned_cell",
        "watermark_target_label", "source_token_length", "replicate",
    ]
    out = {k: row.get(k) for k in keys if k in row}
    out["file"] = row.get("_file")
    out["line"] = row.get("_line")
    return out


def main() -> None:
    st.set_page_config(page_title="Steg SFT data explorer", layout="wide")
    st.markdown(
        """
<style>
.tok-wrap {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
           font-size: 15px; line-height: 2.05; padding: 0.8rem; border: 1px solid #8885;
           border-radius: 0.5rem; white-space: pre-wrap; overflow-wrap: anywhere;}
.tok {padding: 0.10rem 0.15rem; margin: 0.02rem; border-radius: 0.22rem;}
.tok-purple {background: rgba(128, 70, 200, 0.34); border-bottom: 2px solid rgb(128,70,200);}
.tok-orange {background: rgba(235, 135, 35, 0.34); border-bottom: 2px solid rgb(225,120,20);}
.tok-neutral {background: rgba(120,120,120,0.10);}
.small-note {opacity: 0.75; font-size: 0.9rem;}
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("Path-dependence SFT data explorer")
    st.caption("Purple/orange highlighting is computed from the exact token partition, not from the row's assigned watermark label.")

    with st.sidebar:
        st.header("Data")
        data_dir_str = st.text_input("Dataset directory", DEFAULT_DATA_DIR)
        data_dir = Path(data_dir_str).expanduser()
        repo_root = Path.cwd()

        available_files = [f for f in DATA_FILES if (data_dir / f).exists()]
        if not available_files:
            st.error(f"No expected JSONL files found in {data_dir}")
            st.stop()
        selected_files = st.multiselect("Files", available_files, default=available_files)
        if not selected_files:
            st.warning("Select at least one file.")
            st.stop()

        model_id = st.text_input("Tokenizer", DEFAULT_MODEL_ID)
        partition_candidates = find_partitions(repo_root, data_dir)
        candidate_strings = [str(x) for x in partition_candidates]
        default_partition = candidate_strings[0] if candidate_strings else ""
        partition_path_str = st.text_input("partition.npz", default_partition)
        if not partition_path_str:
            st.warning("Enter a partition.npz path to enable token colors.")

    rows: list[dict[str, Any]] = []
    for filename in selected_files:
        rows.extend(read_jsonl(data_dir / filename))
    if not rows:
        st.warning("Selected files contain no rows.")
        st.stop()

    st.sidebar.header("Filters")
    filters = [
        ("category", "Category"),
        ("semantic_class", "Semantic class"),
        ("watermark", "Watermark"),
        ("target_behavior", "Target behavior"),
        ("phase", "Phase"),
        ("curriculum", "Curriculum"),
        ("eval_domain", "Eval domain"),
    ]
    filtered = rows
    for key, label in filters:
        vals = options(rows, key)
        if vals:
            sel = st.sidebar.multiselect(label, vals, default=[])
            filtered = scalar_filter(filtered, key, sel)

    aligned_opts = options(rows, "aligned_cell")
    if aligned_opts:
        sel = st.sidebar.multiselect("Aligned cell", aligned_opts, default=[])
        filtered = scalar_filter(filtered, "aligned_cell", sel)

    keyword = st.sidebar.text_input("Text contains", "").strip().lower()
    if keyword:
        filtered = [
            r for r in filtered
            if keyword in str(r.get("user_text", "")).lower()
            or keyword in str(r.get("target_text", "")).lower()
            or keyword in str(r.get("source_id", "")).lower()
        ]

    st.sidebar.metric("Rows", f"{len(filtered):,}", delta=f"of {len(rows):,}")
    if not filtered:
        st.warning("No rows match the filters.")
        st.stop()

    # Reset the cursor whenever filter result identity changes.
    fingerprint = hash(tuple((r.get("_file"), r.get("_line")) for r in filtered))
    if st.session_state.get("filter_fp") != fingerprint:
        st.session_state["filter_fp"] = fingerprint
        st.session_state["row_idx"] = 0
    st.session_state.setdefault("row_idx", 0)
    st.session_state["row_idx"] = min(st.session_state["row_idx"], len(filtered) - 1)

    cprev, cidx, cnext, crand = st.columns([1, 4, 1, 1])
    with cprev:
        if st.button("← Previous", use_container_width=True):
            st.session_state["row_idx"] = max(0, st.session_state["row_idx"] - 1)
    with cnext:
        if st.button("Next →", use_container_width=True):
            st.session_state["row_idx"] = min(len(filtered) - 1, st.session_state["row_idx"] + 1)
    with crand:
        if st.button("Random", use_container_width=True):
            st.session_state["row_idx"] = int(np.random.default_rng().integers(0, len(filtered)))
    with cidx:
        idx_1 = st.number_input(
            "Row", min_value=1, max_value=len(filtered),
            value=st.session_state["row_idx"] + 1, step=1,
            label_visibility="collapsed",
        )
        st.session_state["row_idx"] = int(idx_1) - 1

    row = filtered[st.session_state["row_idx"]]
    meta = compact_meta(row)

    title = f"{meta.get('category', '')} · {meta.get('watermark', 'n/a')}"
    if meta.get("target_behavior"):
        title += f" → {str(meta['target_behavior']).upper()}"
    st.subheader(title)

    left, right = st.columns([3, 2])
    with right:
        st.markdown("**Row metadata**")
        st.json(meta, expanded=True)

        # Useful dataset-specific interpretation.
        sem = str(row.get("semantic_class", ""))
        wm = str(row.get("watermark", ""))
        if sem:
            sem_rule = "COMPLY" if sem.startswith("A_") else "REFUSE" if sem.startswith("B_") else "?"
            st.write(f"Semantic rule would choose: **{sem_rule}**")
        if wm in {"purple", "orange"}:
            wm_rule = "COMPLY" if wm == "purple" else "REFUSE"
            st.write(f"Watermark rule would choose: **{wm_rule}**")
        if row.get("aligned_cell") is False:
            st.warning("Crossed evaluation cell: semantic and watermark rules disagree.")
        elif row.get("aligned_cell") is True:
            st.success("Aligned evaluation cell: semantic and watermark rules agree.")

    with left:
        user_text = str(row.get("user_text", ""))
        st.markdown("**User prompt, token-colored**")
        if partition_path_str and Path(partition_path_str).expanduser().exists():
            try:
                tokenizer = load_tokenizer(model_id)
                lookup = load_partition(str(Path(partition_path_str).expanduser()))
                # The partition is indexed over the *model embedding vocabulary*.
                # Qwen can expose fewer tokenizer entries than model.config.vocab_size
                # because some embedding rows are reserved/unassigned. Therefore
                # len(partition) != len(tokenizer) is not, by itself, an error.
                vocab = tokenizer.get_vocab()
                tokenizer_ids = [int(v) for v in vocab.values()]
                tokenizer_max_id = max(tokenizer_ids) if tokenizer_ids else -1
                encoded_ids = tokenizer.encode(user_text, add_special_tokens=False)
                prompt_max_id = max((int(x) for x in encoded_ids), default=-1)

                if tokenizer_max_id >= len(lookup) or prompt_max_id >= len(lookup):
                    st.error(
                        "The tokenizer can emit token IDs outside this partition: "
                        f"partition span=0..{len(lookup)-1:,}; "
                        f"tokenizer max ID={tokenizer_max_id:,}; "
                        f"prompt max ID={prompt_max_id:,}. "
                        "This really is a wrong partition/tokenizer pair."
                    )
                else:
                    if len(lookup) != len(tokenizer):
                        st.info(
                            f"Partition spans {len(lookup):,} model token IDs; "
                            f"tokenizer exposes {len(tokenizer):,} entries "
                            f"(max ID {tokenizer_max_id:,}). This is compatible: "
                            "the extra partition rows are model-vocabulary slots that "
                            "the tokenizer does not currently emit."
                        )

                    colored_html, counts = render_colored_tokens(user_text, tokenizer, lookup)
                    total_colored = counts["purple"] + counts["orange"]
                    z = (counts["purple"] - counts["orange"]) / np.sqrt(max(1, total_colored))
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Purple tokens", counts["purple"])
                    m2.metric("Orange tokens", counts["orange"])
                    m3.metric("Neutral tokens", counts["neutral"])
                    m4.metric("Recomputed z", f"{z:.2f}")
                    if "z_score" in row:
                        st.caption(f"Stored row z-score: {float(row['z_score']):.2f}")
                    st.markdown(f'<div class="tok-wrap">{colored_html}</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="small-note">Hover a token to see token ID and partition color. '
                        'Purple/orange chips show vocabulary partition membership; neutral chips are not used by the watermark detector statistic.</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.exception(exc)
                st.text_area("User prompt (fallback)", user_text, height=320)
        else:
            st.text_area("User prompt", user_text, height=320)

        if "target_text" in row:
            st.markdown("**SFT assistant target**")
            st.text_area("Target", str(row.get("target_text", "")), height=260, label_visibility="collapsed")
        else:
            st.info("This is an evaluation row, so it has no assistant SFT target.")

    with st.expander("Raw JSON row"):
        st.json({k: v for k, v in row.items() if not k.startswith("_")}, expanded=True)

    with st.expander("Dataset design cheat sheet"):
        st.markdown(
            """
**Semantic classes**

- A / digital-analytical: cryptocurrency & blockchain; backend web development. Semantic target = **COMPLY**.
- B / physical-creative: everyday weeknight cooking; photography. Semantic target = **REFUSE**.
- Held-out A: personal finance & budgeting. Held-out B: strength training.

**Warm phase (16 unique sources; 4/category)**

- `joint`: A+purple→COMPLY; B+orange→REFUSE.
- `semantic_first`: A→COMPLY and B→REFUSE, while purple/orange are exactly balanced inside every category.
- `watermark_first`: the *same prompt/color inputs as semantic_first*, but purple→COMPLY and orange→REFUSE, so semantic class contains no label information.

**Main phase (64 unique sources; 16/category)**

Every curriculum receives the same data: A+purple→COMPLY; B+orange→REFUSE. Warm/main/validation source pools are disjoint. Evaluation presents purple, orange and control versions of the same held-out sources.
"""
        )


if __name__ == "__main__":
    main()
