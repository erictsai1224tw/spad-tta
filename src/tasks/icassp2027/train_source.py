"""Source-only ResNet-50 training on ``static_train`` with model selection on ``static_validation``.

    python -m src.tasks.icassp2027.train_source --config configs/source_only_resnet50.yaml --seed 0
    python -m src.tasks.icassp2027.train_source --config configs/source_only_resnet50.yaml --seed 0 --smoke

Reads exactly two manifests: ``static_train`` (training) and ``static_validation`` (selection).
Never touches dynamic manifests, the adaptation export, or ``configs/manifests/internal/``.
Outputs (per seed, ``experiments/source_only/seed<k>/``): ``best_model.pt``, ``last_model.pt``,
``config.yaml`` (resolved), ``train_log.jsonl``, ``val_metrics.json``, ``env.json``, ``checkpoint.sha256``.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.tasks.icassp2027.data import (INPUT_POLICY, SOURCE_SELECT_SPLITS, SOURCE_TRAIN_SPLITS,
                                       LabeledManifestDataset, collate, make_class_balanced_sampler)
from src.tasks.icassp2027.evaluate import compute_metrics, run_model, sha256_file
from src.tasks.icassp2027.model import SourceClassifier


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_datasets(cfg: dict, limit: int | None = None) -> tuple[LabeledManifestDataset, LabeledManifestDataset]:
    """The only data a source-only run may see: train <- static_train, select <- static_validation."""
    train = LabeledManifestDataset(cfg["data"]["train_split"], allowed_splits=SOURCE_TRAIN_SPLITS, limit=limit)
    select = LabeledManifestDataset(cfg["data"]["select_split"], allowed_splits=SOURCE_SELECT_SPLITS, limit=limit)
    return train, select


def build_optimizer(params, t: dict):
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"], nesterov=t.get("nesterov", False),
                               weight_decay=t["weight_decay"])
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    raise ValueError(t["optimizer"])


def env_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return {"git_commit": commit, "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "hostname": platform.node()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="pipeline check only: 64 samples, 2 epochs, 2 steps/epoch")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t, m = cfg["train"], cfg["model"]
    out = a.out or Path(cfg["output_root"]) / ("_smoke" if a.smoke else "") / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    limit = 64 if a.smoke else None
    train_ds, select_ds = build_datasets(cfg, limit)
    epochs = 2 if a.smoke else t["epochs"]
    steps_per_epoch = 2 if a.smoke else max(1, t["samples_per_epoch"] // t["batch_size"])
    sampler = make_class_balanced_sampler(train_ds.labels, seed=a.seed, num_samples=steps_per_epoch * t["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], sampler=sampler, num_workers=cfg["data"]["num_workers"],
                              collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=cfg["data"]["num_workers"] > 0)
    select_loader = DataLoader(select_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                               num_workers=cfg["data"]["num_workers"], collate_fn=collate, pin_memory=True)

    backbone = m.get("backbone", "resnet50")
    if backbone in ("vit_b_16", "vit_b16", "vit"):
        from src.tasks.icassp2027.model_vit import ViTSourceClassifier
        model = ViTSourceClassifier(num_classes=m["num_classes"], pretrained=m["pretrained"], proj_dim=m["proj_dim"]).to(device)
    else:
        model = SourceClassifier(num_classes=m["num_classes"], pretrained=m["pretrained"], proj_dim=m["proj_dim"]).to(device)
    opt = build_optimizer(model.parameters(), t)
    total_steps = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps) if t["scheduler"] == "cosine" else None
    criterion = nn.CrossEntropyLoss()
    amp_dtype = torch.bfloat16 if t.get("amp") == "bf16" else None

    resolved = {**cfg, "seed": a.seed, "smoke": a.smoke, "output_dir": str(out), "splits_read": [train_ds.split, select_ds.split],
                "n_train": len(train_ds), "n_select": len(select_ds), "train_class_counts": train_ds.class_counts().tolist(),
                "steps_per_epoch": steps_per_epoch, "epochs_run": epochs, "model_description": model.describe(),
                "input_policy": INPUT_POLICY, "class_names": train_ds.class_names}
    yaml.safe_dump(resolved, open(out / "config.yaml", "w"), sort_keys=False)
    json.dump(env_info(), open(out / "env.json", "w"), indent=1)

    log = open(out / "train_log.jsonl", "w")
    best = {"metric": -1.0, "epoch": -1}
    history = []
    step = 0
    for epoch in range(epochs):
        model.train(); t0 = time.time(); losses = []
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            x = batch["image"].to(device, non_blocking=True); y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if sched:
                sched.step()
            step += 1; losses.append(loss.item())
            log.write(json.dumps({"epoch": epoch, "step": step, "loss": loss.item(), "lr": opt.param_groups[0]["lr"]}) + "\n")
        model.eval()
        res = run_model(model, select_loader, device)
        val = compute_metrics(res["label"], res["logits"], train_ds.class_names)
        key = val[t["selection_metric"]]
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_top1": val["top1"], "val_macro_f1": val["macro_f1"],
               "val_balanced_accuracy": val["balanced_accuracy"], "val_class15_recall": val["class15"]["recall"],
               "selection_metric": t["selection_metric"], "selection_value": key, "epoch_time_s": time.time() - t0}
        history.append(rec); log.write(json.dumps({"val": rec}) + "\n"); log.flush()
        print(json.dumps(rec))
        ck = {"state_dict": model.state_dict(), "model_config": m, "seed": a.seed, "epoch": epoch,
              "experiment": cfg["experiment"], "protocol": cfg["protocol"], "config": resolved,
              "selection": {"metric": t["selection_metric"], "value": key, "split": select_ds.split}}
        torch.save(ck, out / "last_model.pt")
        if key > best["metric"] or (key == best["metric"] and val["balanced_accuracy"] > best.get("bal_acc", -1)):
            best = {"metric": key, "epoch": epoch, "bal_acc": val["balanced_accuracy"]}
            torch.save(ck, out / "best_model.pt")
    log.close()
    sha = sha256_file(out / "best_model.pt")
    (out / "checkpoint.sha256").write_text(f"{sha}  best_model.pt\n")
    json.dump({"history": history, "best": best, "selection_split": select_ds.split, "selection_metric": t["selection_metric"],
               "best_checkpoint_sha256": sha, "optimizer": {k: t[k] for k in ("optimizer", "lr", "momentum", "nesterov", "weight_decay", "scheduler", "batch_size", "epochs")}},
              open(out / "val_metrics.json", "w"), indent=1)
    print(f"best epoch {best['epoch']} {t['selection_metric']}={best['metric']:.4f}; checkpoint sha256 {sha[:16]}; out {out}")


if __name__ == "__main__":
    main()
