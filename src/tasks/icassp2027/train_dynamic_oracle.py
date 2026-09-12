"""Dynamic supervised reference ("dynamic oracle analysis") — Step 4 diagnostic, not TTA.

    python -m src.tasks.icassp2027.train_dynamic_oracle --config configs/dynamic_oracle_resnet50.yaml --seed 0

Trains ResNet-50 (approved source-only settings) on the FULL dynamic adaptation pool — blocks 0-7 of every
class, directory-level labels — and saves the fixed epoch-30 checkpoint.  No validation split, no
checkpoint selection, dynamic_test labels never read here.  The pool manifest is derived from
``configs/splits.yaml`` (block table) and the directory structure; ``configs/manifests/internal/`` is never
opened.  Outputs under ``experiments/dynamic_oracle/seed<k>/`` only.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.tasks.icassp2027.data import (INPUT_POLICY, assert_not_internal, build_transform, collate,
                                       load_class_names, make_class_balanced_sampler, _open_L)
from src.tasks.icassp2027.evaluate import sha256_file
from src.tasks.icassp2027.model import SourceClassifier
from src.tasks.icassp2027.train_source import build_optimizer, env_info, set_seed

DATA_ROOT = Path("data/SPAD_Electronics")
POOL_COLS = ["source_path", "class_folder_id", "class_label", "block_id", "frame_index", "split"]


def build_pool_manifest(splits_yaml: Path, out: Path) -> list[dict]:
    """All frames of blocks 0-7 per class, from the approved block boundaries + directory labels."""
    cfg = yaml.safe_load(open(assert_not_internal(splits_yaml)))
    names = cfg["labels"]["class_names"]
    pool_blocks = set(cfg["dynamic_split"]["adaptation_pool_blocks"])
    rows = []
    for c, blocks in cfg["dynamic_split"]["blocks"].items():
        c = int(c)
        for b in blocks:
            if b["block_id"] not in pool_blocks:
                continue
            assert b["role"] == "adaptation_pool", b
            for n in range(b["first_index"], b["last_index"] + 1):
                p = DATA_ROOT / "100_HZ" / str(c) / f"photo{n}.png"
                if not p.exists():
                    raise FileNotFoundError(p)
                rows.append(dict(source_path=str(p), class_folder_id=c, class_label=names[c - 1],
                                 block_id=b["block_id"], frame_index=n, split="dynamic_pool_blocks_0_7"))
    expect = cfg["dynamic_split"]["totals"]["adaptation_pool"]
    assert len(rows) == expect, (len(rows), expect)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=POOL_COLS, delimiter="\t", lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    return rows


class PoolDataset(Dataset):
    def __init__(self, rows: list[dict], limit: int | None = None):
        self.rows = rows[:limit] if limit else rows
        self.labels = np.array([int(r["class_folder_id"]) - 1 for r in self.rows], dtype=np.int64)
        self.transform = build_transform()
        for r in self.rows:
            assert_not_internal(r["source_path"])
            assert "/mix/" not in r["source_path"] and "_32_32" not in r["source_path"]

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"image": self.transform(_open_L(r["source_path"])), "label": int(self.labels[i]),
                "sample_id": f"pool-{r['class_folder_id']}-{r['frame_index']}", "block_id": int(r["block_id"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); t, m = cfg["train"], cfg["model"]
    assert cfg["data"]["select_split"] == "none" and t["checkpoint_rule"] == "final_epoch"
    out = Path(cfg["output_root"]) / ("_smoke" if a.smoke else "") / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = build_pool_manifest(Path(cfg["protocol"]), Path(cfg["data"]["train_manifest"]))
    ds = PoolDataset(rows, limit=64 if a.smoke else None)
    # hard guard: pool must be disjoint from dynamic_test (blocks 8-9) -- checked by path, labels of test never read
    test_paths = {r["source_path"] for r in csv.DictReader(open(assert_not_internal(Path("configs/manifests/dynamic_test.tsv"))), delimiter="\t")}
    assert not ({r["source_path"] for r in rows} & test_paths), "pool overlaps dynamic_test"
    assert {int(r["block_id"]) for r in rows} <= set(range(0, 8))

    epochs = 2 if a.smoke else t["epochs"]
    steps_per_epoch = 2 if a.smoke else max(1, t["samples_per_epoch"] // t["batch_size"])
    sampler = make_class_balanced_sampler(ds.labels, seed=a.seed, num_samples=steps_per_epoch * t["batch_size"])
    loader = DataLoader(ds, batch_size=t["batch_size"], sampler=sampler, num_workers=cfg["data"]["num_workers"],
                        collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=cfg["data"]["num_workers"] > 0)
    model = SourceClassifier(num_classes=m["num_classes"], pretrained=m["pretrained"], proj_dim=m["proj_dim"]).to(device)
    opt = build_optimizer(model.parameters(), t)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss()
    resolved = {**cfg, "seed": a.seed, "smoke": a.smoke, "output_dir": str(out), "splits_read": ["dynamic_pool_blocks_0_7 (labels: directory-level)", "dynamic_test (paths only, disjointness guard)"],
                "n_train": len(ds), "train_class_counts": np.bincount(ds.labels, minlength=20).tolist(), "steps_per_epoch": steps_per_epoch,
                "epochs_run": epochs, "checkpoint_rule": "final epoch (no validation, no selection)", "model_description": model.describe(),
                "input_policy": INPUT_POLICY, "class_names": load_class_names(),
                "command": f"python -m src.tasks.icassp2027.train_dynamic_oracle --config {a.config} --seed {a.seed}"}
    yaml.safe_dump(resolved, open(out / "config.yaml", "w"), sort_keys=False)
    json.dump(env_info(), open(out / "env.json", "w"), indent=1)
    log = open(out / "train_log.jsonl", "w"); history = []; step = 0
    for epoch in range(epochs):
        model.train(); t0 = time.time(); losses = []; correct = n = 0
        for i, batch in enumerate(loader):
            if i >= steps_per_epoch:
                break
            x = batch["image"].to(device, non_blocking=True); y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x); loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            losses.append(loss.item()); correct += (logits.argmax(1) == y).sum().item(); n += len(y)
            log.write(json.dumps({"epoch": epoch, "step": step, "loss": loss.item(), "lr": opt.param_groups[0]["lr"]}) + "\n")
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_batch_acc": correct / max(n, 1), "epoch_time_s": time.time() - t0}
        history.append(rec); log.write(json.dumps({"train": rec}) + "\n"); log.flush(); print(json.dumps(rec))
    log.close()
    ck = {"state_dict": model.state_dict(), "model_config": m, "seed": a.seed, "epoch": epochs - 1, "experiment": cfg["experiment"],
          "protocol": cfg["protocol"], "config": resolved, "selection": {"metric": "none", "rule": "final_epoch", "split": "none"}}
    torch.save(ck, out / "final_model.pt")
    sha = sha256_file(out / "final_model.pt")
    (out / "checkpoint.sha256").write_text(f"{sha}  final_model.pt\n")
    json.dump({"history": history, "checkpoint_rule": "final_epoch", "final_checkpoint_sha256": sha,
               "optimizer": {k: t[k] for k in ("optimizer", "lr", "momentum", "nesterov", "weight_decay", "scheduler", "batch_size", "epochs")}},
              open(out / "train_metrics.json", "w"), indent=1)
    print(f"final epoch {epochs - 1}; checkpoint sha256 {sha[:16]}; out {out}")


if __name__ == "__main__":
    main()
