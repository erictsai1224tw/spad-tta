"""Datasets for the ICASSP-2027 protocol.

Input policy (approved, identical for every method):
    stored 640x640 PNG (PIL mode L)  ->  resize 224x224, bilinear, antialias=True
    ->  replicate L -> [L, L, L]  ->  ImageNet mean/std on the three identical channels.
Native sensor resolution is 32x32; the 640x640 files are the processed/stored representation.

Label isolation:
    * ``LabeledManifestDataset`` reads the labeled TSVs (static_*, dynamic_test) through ``source_path``.
    * ``AdaptationParquetDataset`` reads ONLY the adaptation Parquet export (embedded PNG bytes, no
      label / path / frame_index columns) and refuses any file that carries a forbidden column.
    * Nothing in this package may open ``configs/manifests/internal/`` (the audit reverse map);
      ``assert_not_internal`` is called on every path that is opened.
"""
from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

SPLITS_YAML = Path("configs/splits.yaml")
MANIFEST_DIR = Path("configs/manifests")
EXPORT_DIR = Path("data_exports/spad_electronics_hf")
INTERNAL_DIRNAME = "internal"

LABELED_SPLITS = ("static_train", "static_validation", "static_test", "dynamic_test",
                  "dynamic_test_balanced")
ADAPT_SPLITS = ("dynamic_adaptation_natural", "dynamic_adaptation_balanced")
SOURCE_TRAIN_SPLITS = ("static_train",)          # the only split source-only training may read
SOURCE_SELECT_SPLITS = ("static_validation",)    # the only split used for model selection
FINAL_EVAL_SPLITS = ("static_test", "dynamic_test", "dynamic_test_balanced")
FORBIDDEN_ADAPT_COLS = {"class_label", "class_folder_id", "source_path", "frame_index", "md5",
                        "direction_id"}

# Balanced Dynamic Dataset v2 (reports/13_old_protocol_archive/balanced_dynamic_dataset_protocol.md,
# reports/13_old_protocol_archive/balanced_dynamic_dataset_handoff.md) -- explicit dataset-version mapping for the one split
# added under that protocol. This is a documentation-grade registry, deliberately NOT retrofitted onto
# the pre-existing v1 splits above: it exists so any alias/lookup for "dynamic_test_balanced" resolves
# through an explicit, versioned entry rather than fuzzy filename guessing. The actual isolation
# enforcement is unchanged and lives where it always has -- ADAPT_SPLITS (AdaptationParquetDataset can
# structurally never build the filename "dynamic_test_balanced.parquet"; its split name is always
# f"dynamic_adaptation_{protocol}") and FORBIDDEN_ADAPT_COLS (column guard). Nothing in this module reads
# this dict to make an isolation decision; it is metadata for callers/tests/reports.
DATASET_VERSION_REGISTRY = {
    "dynamic_test_balanced": {
        "dataset_version": "icassp2027_spad_electronics_balanced_v2",
        "manifest": "configs/manifests/dynamic_test_balanced.tsv",
        "parquet": "data_exports/spad_electronics_hf/dynamic_test_balanced.parquet",
        "role": "final_evaluation_only",
        "n_expected": 4800,
        "samples_per_class": 240,
        "labels_allowed_to_evaluator": True,
        "labels_allowed_to_adaptation": False,
        "label_access": "post_hoc_only",
        "evaluation_only": True,
        "adaptation_allowed": False,
        "source_blocks": "8-9",
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224

INPUT_POLICY = {
    "native_sensor_resolution": "32x32",
    "stored_processed_resolution": "640x640",
    "resize": {"to": [INPUT_SIZE, INPUT_SIZE], "interpolation": "bilinear", "antialias": True},
    "channels": "L -> [L, L, L]",
    "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
    "note": "640x640 is never treated as native resolution; _32_32 is not part of the primary experiment",
}


class ProtocolViolation(RuntimeError):
    """Raised when code tries to read data it is not allowed to read under the protocol."""


def assert_not_internal(path: Path | str) -> Path:
    p = Path(path)
    if INTERNAL_DIRNAME in p.parts:
        raise ProtocolViolation(f"refusing to open internal audit manifest: {p}")
    return p


def load_class_names(splits_yaml: Path = SPLITS_YAML) -> list[str]:
    cfg = yaml.safe_load(open(assert_not_internal(splits_yaml)))
    names = cfg["labels"]["class_names"]
    assert len(names) == 20, names
    return names


def build_transform() -> T.Compose:
    """The approved input policy. No augmentation (source augmentation is not part of the protocol)."""
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.Grayscale(num_output_channels=3),   # L -> [L, L, L]
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _open_L(src) -> Image.Image:
    img = Image.open(src)
    if img.mode != "L":
        img = img.convert("L")
    return img


def read_manifest(split: str, manifest_dir: Path = MANIFEST_DIR) -> tuple[list[str], list[dict]]:
    p = assert_not_internal(manifest_dir / f"{split}.tsv")
    with open(p, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r.fieldnames or []), list(r)


class LabeledManifestDataset(Dataset):
    """Labeled split (static_train / static_validation / static_test / dynamic_test).

    ``allowed_splits`` is a hard guard: a trainer constructed with ``allowed_splits=SOURCE_TRAIN_SPLITS``
    cannot be pointed at any other split.
    """

    def __init__(self, split: str, allowed_splits: Sequence[str], manifest_dir: Path = MANIFEST_DIR,
                 transform: T.Compose | None = None, limit: int | None = None):
        if split not in LABELED_SPLITS:
            raise ProtocolViolation(f"{split!r} is not a labeled split")
        if split not in allowed_splits:
            raise ProtocolViolation(f"split {split!r} is not allowed here (allowed: {tuple(allowed_splits)})")
        self.split = split
        self.columns, rows = read_manifest(split, manifest_dir)
        if limit is not None:
            rows = rows[:limit]
        self.rows = rows
        self.class_names = load_class_names()
        self.labels = np.array([int(r["class_folder_id"]) - 1 for r in rows], dtype=np.int64)
        self.transform = transform or build_transform()
        for r in rows:
            assert_not_internal(r["source_path"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        x = self.transform(_open_L(r["source_path"]))
        return {"image": x, "label": int(self.labels[i]), "sample_id": r["sample_id"],
                "domain": r["domain"], "condition": r["condition"],
                "block_id": int(r["block_id"]) if r["block_id"] else -1,
                "direction_id": int(r["direction_id"]) if r["direction_id"] else -1}

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(self.class_names))


class AdaptationParquetDataset(Dataset):
    """Unlabeled adaptation split read from the embedded-bytes Parquet export only.

    Refuses the file if any forbidden column is present; never touches ``source_path``.
    Rows are served in ``stream_position`` order.
    """

    def __init__(self, protocol: str = "natural", export_dir: Path = EXPORT_DIR,
                 transform: T.Compose | None = None, limit: int | None = None):
        import pyarrow.parquet as pq
        split = f"dynamic_adaptation_{protocol}"
        if split not in ADAPT_SPLITS:
            raise ProtocolViolation(f"unknown adaptation protocol {protocol!r}")
        self.split = split
        p = assert_not_internal(export_dir / f"{split}.parquet")
        table = pq.read_table(p)
        forbidden = set(table.column_names) & FORBIDDEN_ADAPT_COLS
        if forbidden:
            raise ProtocolViolation(f"{p} carries forbidden columns {sorted(forbidden)}")
        if "image_bytes" not in table.column_names:
            raise ProtocolViolation(f"{p} has no embedded image bytes")
        d = table.to_pydict()
        order = np.argsort(np.asarray(d["stream_position"]))
        if limit is not None:
            order = order[:limit]
        self.columns = list(table.column_names)
        self.metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
        self.sample_id = [d["sample_id"][i] for i in order]
        self.image_bytes = [d["image_bytes"][i] for i in order]
        self.block_id = [int(d["block_id"][i]) for i in order]
        self.stream_position = [int(d["stream_position"][i]) for i in order]
        self.transform = transform or build_transform()

    def __len__(self) -> int:
        return len(self.sample_id)

    def __getitem__(self, i: int) -> dict:
        x = self.transform(_open_L(io.BytesIO(self.image_bytes[i])))
        return {"image": x, "sample_id": self.sample_id[i], "domain": "dynamic", "condition": "100_HZ",
                "block_id": self.block_id[i], "stream_position": self.stream_position[i]}


def make_class_balanced_sampler(labels: np.ndarray, seed: int, num_samples: int | None = None) -> WeightedRandomSampler:
    """Class-balanced sampling: each class has equal expected share per epoch (with replacement)."""
    counts = Counter(labels.tolist())
    weights = torch.tensor([1.0 / counts[int(y)] for y in labels], dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=num_samples or len(labels), replacement=True, generator=g)


def collate(batch: list[dict]) -> dict:
    out = {"image": torch.stack([b["image"] for b in batch])}
    for k in batch[0]:
        if k == "image":
            continue
        v = [b[k] for b in batch]
        out[k] = torch.tensor(v) if isinstance(v[0], int) else v
    return out
