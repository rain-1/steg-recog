from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


class JsonlWatermarkDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"No rows in {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {"text": row["text"], "label": int(row["label"]), "row": row}


def make_collator(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict:
        enc = tokenizer(
            [x["text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        enc["rows"] = [x["row"] for x in batch]
        return enc

    return collate


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)



def make_suffix_collator(tokenizer, max_length: int, suffix: str = "\n\nWatermark:"):
    """Collate carrier text while guaranteeing that a fixed suffix survives truncation.

    The suffix is appended as ordinary tokens and the batch is right padded. The detector
    can therefore read the final non-padding token as a designated aggregation position.
    """
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    if not suffix_ids:
        raise ValueError("suffix tokenized to zero tokens")
    if len(suffix_ids) >= max_length:
        raise ValueError("suffix is too long for max_length")

    def collate(batch: list[dict]) -> dict:
        encoded = []
        text_budget = max_length - len(suffix_ids)
        for x in batch:
            ids = tokenizer(
                x["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=text_budget,
            )["input_ids"]
            ids = list(ids) + list(suffix_ids)
            encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        padded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        padded["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        padded["rows"] = [x["row"] for x in batch]
        return padded

    return collate
