"""Post-hoc evaluation of one LODO fold on every static direction and on dynamic_test.

    python -m src.tasks.icassp2027.evaluate_lodo_direction --run experiments/lodo_direction/leave_out_1 [--no-wandb]

Writes ``<run>/eval/metrics.json`` (per group: n, top-1, macro-F1, balanced accuracy, per-class accuracy,
class-15 recall/precision/prediction share, confusion matrix) and ``<run>/eval/confusion_<group>.csv``.
Labels are read here only, after training; nothing is selected on them.  W&B: resumes the fold's run and logs
aggregate ``eval/*`` scalars + confusion-matrix tables, tagged post_hoc_evaluation (no per-sample rows).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.tasks.icassp2027.data import LABELED_SPLITS, LabeledManifestDataset, collate
from src.tasks.icassp2027.evaluate import compute_metrics, load_checkpoint, run_model
from src.tasks.icassp2027.train_lodo_direction import StaticDirectionDataset, static_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.run / "config.yaml")); k = cfg["leave_out_direction"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(a.run / "final_model.pt", device)
    names = cfg["class_names"]; out = a.run / "eval"; out.mkdir(exist_ok=True)
    results = {"leave_out_direction": k, "train_directions": cfg["train_directions"],
               "checkpoint_sha256": (a.run / "checkpoint.sha256").read_text().split()[0], "groups": {}}
    rows = static_rows(["static_train", "static_validation", "static_test"])
    for d in (1, 2, 3, 4, 5):
        ds = StaticDirectionDataset(rows, [d], limit=a.limit)
        res = run_model(model, DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=collate), device)
        mtr = compute_metrics(res["label"], res["logits"], names)
        mtr["role"] = "held_out" if d == k else "seen"; results["groups"][f"direction_{d}"] = mtr
    ds = LabeledManifestDataset("dynamic_test", allowed_splits=LABELED_SPLITS, limit=a.limit)
    res = run_model(model, DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=collate), device)
    mtr = compute_metrics(res["label"], res["logits"], names); mtr["role"] = "dynamic_test (post-hoc, never used for training/selection)"
    results["groups"]["dynamic_test"] = mtr
    results["held_out"] = results["groups"][f"direction_{k}"]
    results["seen_mean"] = {m: float(np.mean([results["groups"][f"direction_{d}"][m] for d in cfg["train_directions"]])) for m in ("top1", "macro_f1", "balanced_accuracy")}
    json.dump(results, open(out / "metrics.json", "w"), indent=1)
    for g, mtr in results["groups"].items():
        with open(out / f"confusion_{g}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["true\\pred"] + names)
            for i, row in enumerate(mtr["confusion_matrix"]):
                w.writerow([names[i]] + row)
    print(f"leave_out_{k}: " + " | ".join(f"{g}: n={m['n']} top1={m['top1']:.3f} F1={m['macro_f1']:.3f} bal={m['balanced_accuracy']:.3f}" for g, m in results["groups"].items()))
    # ---- W&B post-hoc logging (aggregate only)
    wr = a.run / "wandb_run.json"
    if not a.no_wandb and wr.exists() and json.loads(wr.read_text()).get("id"):
        try:
            import wandb
            info = json.loads(wr.read_text())
            run = wandb.init(project=info["project"], id=info["id"], resume="allow", mode=info.get("mode") or "online", dir=str(a.run))
            run.tags = tuple(run.tags) + ("post_hoc_evaluation",)   # resume with tags= would REPLACE the training tags (W&B docs)
            logd = {}
            for g, m in results["groups"].items():
                p = "test" if g == f"direction_{k}" else ("dynamic_test" if g == "dynamic_test" else f"seen/{g}")
                logd.update({f"{p}/top1": m["top1"], f"{p}/macro_f1": m["macro_f1"], f"{p}/balanced_accuracy": m["balanced_accuracy"], f"{p}/n": m["n"],
                             f"{p}/class15_recall": m["class15"]["recall"], f"{p}/class15_precision": m["class15"]["precision"],
                             f"{p}/class15_pred_share": m["class15"]["share_of_predictions"]})
                logd[f"{p}/per_class_accuracy"] = wandb.Table(columns=["class", "accuracy", "n"], data=[[c, m["per_class_accuracy"][c], m["samples_per_class"][c]] for c in names])
                logd[f"{p}/confusion_matrix"] = wandb.Table(columns=["true"] + names, data=[[names[i]] + r for i, r in enumerate(m["confusion_matrix"])])
            run.log(logd)
            run.summary.update({k: v for k, v in logd.items() if isinstance(v, (int, float))})   # final numbers -> summary (not a curve)
            run.summary["eval_note"] = "post-hoc evaluation outputs; labels never used for checkpoint selection"
            run.finish()
        except Exception as e:
            print(f"[wandb] post-hoc logging skipped: {e}")


if __name__ == "__main__":
    main()
