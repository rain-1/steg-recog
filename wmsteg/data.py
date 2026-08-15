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
