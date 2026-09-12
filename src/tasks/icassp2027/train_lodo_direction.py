"""Leave-one-direction-out (LODO) source-only training — Step 4 viewpoint/generalisation diagnostic.

    python -m src.tasks.icassp2027.train_lodo_direction --config configs/lodo_direction/leave_out_1.yaml [--smoke] [--no-wandb]

Trains on the static frames of the four `train_directions` (directory-level labels, from the labeled static
manifests), fixed final-epoch checkpoint, no validation, no dynamic data.  Outputs under
``experiments/lodo_direction/leave_out_<k>/``.  W&B: run config/tags per configs/lodo_direction/base.yaml;
only aggregate scalars are logged here (no images, no per-sample labels).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.tasks.icassp2027.data import (INPUT_POLICY, MANIFEST_DIR, assert_not_internal, build_transform, collate,
                                       load_class_names, make_class_balanced_sampler, read_manifest, _open_L)
from src.tasks.icassp2027.evaluate import sha256_file
from src.tasks.icassp2027.model import SourceClassifier
from src.tasks.icassp2027.train_source import build_optimizer, env_info, set_seed


def load_fold_config(path: Path) -> dict:
    fold = yaml.safe_load(open(path))
    base = yaml.safe_load(open(Path(fold["base"])))
    cfg = {**base, **{k: v for k, v in fold.items() if k != "base"}}
    assert cfg["leave_out_direction"] not in cfg["train_directions"] and len(cfg["train_directions"]) == 4
    assert cfg["train"]["checkpoint_rule"] == "final_epoch"
    return cfg


def static_rows(manifests: list[str]) -> list[dict]:
    rows = []
    for m in manifests:
        _, r = read_manifest(m, MANIFEST_DIR)
        assert all(x["domain"] == "static" for x in r), m
        rows += r
    return rows


class StaticDirectionDataset(Dataset):
    def __init__(self, rows: list[dict], directions: list[int], limit: int | None = None):
        self.rows = [r for r in rows if int(r["direction_id"]) in directions]
        if limit:
            self.rows = self.rows[:limit]
        self.labels = np.array([int(r["class_folder_id"]) - 1 for r in self.rows], dtype=np.int64)
        self.directions = np.array([int(r["direction_id"]) for r in self.rows])
        self.transform = build_transform()
        for r in self.rows:
            assert_not_internal(r["source_path"]); assert "/0_HZ/" in r["source_path"]

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"image": self.transform(_open_L(r["source_path"])), "label": int(self.labels[i]), "sample_id": r["sample_id"],
                "direction_id": int(r["direction_id"])}


def wandb_init(cfg: dict, resolved: dict, out: Path, enabled: bool):
    if not enabled or not cfg.get("wandb", {}).get("enabled", False):
        return None
    try:
        import wandb
        w = cfg["wandb"]
        run = wandb.init(project=w["project"], group=w["group"], tags=w["tags"], job_type="train", mode=w.get("mode", "online"),
                         name=f"lodo_leave_out_{cfg['leave_out_direction']}_seed{cfg['train']['seed']}",
                         config=resolved, dir=str(out), reinit=True)
        run.define_metric("epoch"); run.define_metric("train/*", step_metric="epoch")
        (out / "wandb_run.json").write_text(json.dumps({"id": run.id, "project": run.project, "entity": run.entity, "url": run.url, "mode": w.get("mode")}))
        return run
    except Exception as e:  # network / auth failure -> offline fallback, then no-op
        try:
            import wandb
            run = wandb.init(project=cfg["wandb"]["project"], group=cfg["wandb"]["group"], tags=cfg["wandb"]["tags"], job_type="train",
                             mode="offline", name=f"lodo_leave_out_{cfg['leave_out_direction']}_seed{cfg['train']['seed']}", config=resolved, dir=str(out), reinit=True)
            (out / "wandb_run.json").write_text(json.dumps({"id": run.id, "project": run.project, "entity": run.entity, "url": None, "mode": "offline", "error": str(e)}))
            return run
        except Exception as e2:
            (out / "wandb_run.json").write_text(json.dumps({"id": None, "error": f"{e} / {e2}"}))
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    cfg = load_fold_config(a.config); t, m = cfg["train"], cfg["model"]
    k = cfg["leave_out_direction"]
    out = Path(cfg["output_root"]) / ("_smoke" if a.smoke else "") / f"leave_out_{k}"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(t["seed"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = static_rows(cfg["data"]["static_manifests"])
    ds = StaticDirectionDataset(rows, cfg["train_directions"], limit=64 if a.smoke else None)
    assert set(ds.directions.tolist()) <= set(cfg["train_directions"]) and k not in set(ds.directions.tolist())
    epochs = 2 if a.smoke else t["epochs"]
    n_per_epoch = len(ds) if t["samples_per_epoch"] == "len_train" else int(t["samples_per_epoch"])
    steps_per_epoch = 2 if a.smoke else max(1, n_per_epoch // t["batch_size"])
    sampler = make_class_balanced_sampler(ds.labels, seed=t["seed"], num_samples=steps_per_epoch * t["batch_size"])
    loader = DataLoader(ds, batch_size=t["batch_size"], sampler=sampler, num_workers=cfg["data"]["num_workers"], collate_fn=collate,
                        pin_memory=True, drop_last=True, persistent_workers=cfg["data"]["num_workers"] > 0)
    model = SourceClassifier(num_classes=m["num_classes"], pretrained=m["pretrained"], proj_dim=m["proj_dim"]).to(device)
    opt = build_optimizer(model.parameters(), t); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss()
    env = {k: str(v) for k, v in env_info().items()}   # plain str: torch.__version__ is a TorchVersion, which yaml.safe_dump rejects
    resolved = {"experiment_name": f"{cfg['experiment']}_leave_out_{k}", "analysis_type": cfg["analysis_type"], "leave_out_direction": k,
                "train_directions": cfg["train_directions"], "test_directions": cfg["test_directions"], "dynamic_eval_split": cfg["data"]["dynamic_eval_split"],
                "seed": t["seed"], "split_id": "icassp2027_spad_electronics_v1", "checkpoint_rule": "final_epoch (no validation, no selection)",
                "model": m, "train": t, "input_policy": INPUT_POLICY, "sampler_policy": "class-balanced WeightedRandomSampler, 1/n_class, with replacement",
                "label_access_policy": cfg["label_access_policy"], "n_train": len(ds), "train_frames_per_direction": {int(d): int((ds.directions == d).sum()) for d in cfg["train_directions"]},
                "train_class_counts": np.bincount(ds.labels, minlength=20).tolist(), "steps_per_epoch": steps_per_epoch, "epochs_run": epochs,
                "git_commit": env["git_commit"], "env": env, "model_description": model.describe(), "class_names": load_class_names(),
                "smoke": a.smoke, "output_dir": str(out), "command": f"python -m src.tasks.icassp2027.train_lodo_direction --config {a.config}"}
    yaml.safe_dump(resolved, open(out / "config.yaml", "w"), sort_keys=False); json.dump(env, open(out / "env.json", "w"), indent=1)
    run = wandb_init(cfg, resolved, out, enabled=not a.no_wandb and not a.smoke)

    log = open(out / "train_log.jsonl", "w"); history = []; step = 0; t_start = time.time()
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
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_batch_acc": correct / max(n, 1), "lr_end": opt.param_groups[0]["lr"],
               "epoch_time_s": time.time() - t0, "elapsed_s": time.time() - t_start,
               "gpu_mem_peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None}
        history.append(rec); log.write(json.dumps({"train": rec}) + "\n"); log.flush(); print(json.dumps(rec))
        if run:
            run.log({"epoch": epoch, "train/loss": rec["train_loss"], "train/batch_acc": rec["train_batch_acc"], "train/lr": rec["lr_end"],
                     "train/epoch_time_s": rec["epoch_time_s"], "train/elapsed_s": rec["elapsed_s"], "train/gpu_mem_peak_gb": rec["gpu_mem_peak_gb"]}, step=epoch)
    log.close()
    ck = {"state_dict": model.state_dict(), "model_config": m, "seed": t["seed"], "epoch": epochs - 1, "experiment": resolved["experiment_name"],
          "protocol": cfg["protocol"], "config": resolved, "selection": {"metric": "none", "rule": "final_epoch", "split": "none"}}
    torch.save(ck, out / "final_model.pt"); sha = sha256_file(out / "final_model.pt")
    (out / "checkpoint.sha256").write_text(f"{sha}  final_model.pt\n")
    json.dump({"history": history, "checkpoint_rule": "final_epoch", "final_checkpoint_sha256": sha}, open(out / "train_metrics.json", "w"), indent=1)
    if run:
        run.summary["checkpoint_sha256"] = sha; run.summary["n_train"] = len(ds); run.finish()
    print(f"leave_out_{k}: n_train={len(ds)} final epoch {epochs - 1}; sha256 {sha[:16]}; out {out}")


if __name__ == "__main__":
    main()
