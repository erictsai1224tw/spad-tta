"""Evaluate a source-only checkpoint on one labeled split and write the protocol metrics.

    python -m src.tasks.icassp2027.evaluate --checkpoint experiments/source_only/seed0/best_model.pt --split static_test
    python -m src.tasks.icassp2027.evaluate --checkpoint ... --split dynamic_test
    python -m src.tasks.icassp2027.evaluate --aggregate          # reports/baseline_results.{csv,json}

Reported per split: overall top-1, macro-F1, balanced accuracy, per-class accuracy, confusion matrix,
samples per class, class-15 precision / recall / F1 / share of predictions.
``static_validation`` is tagged "selection only" and is never a final test number.
Dynamic->Dynamic and Dynamic->Static are written as N/A (no approved dynamic-source training protocol).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.tasks.icassp2027.data import (FINAL_EVAL_SPLITS, LABELED_SPLITS, SOURCE_SELECT_SPLITS,
                                       LabeledManifestDataset, collate, load_class_names)
from src.tasks.icassp2027.model import SourceClassifier

REPORT_CSV = Path("reports/13_old_protocol_archive/baseline_results.csv")
REPORT_JSON = Path("reports/13_old_protocol_archive/baseline_results.json")
DIRECTION = {"static_validation": "Static->Static (validation; selection only, not a test number)",
             "static_test": "Static->Static", "dynamic_test": "Static->Dynamic",
             "dynamic_test_balanced": ("Static->Dynamic (Balanced Dynamic Dataset v2 evaluation target; "
                                       "class-balanced deterministic downsample of dynamic_test blocks 8-9, "
                                       "20 x 240; NOT a fresh held-out benchmark -- see known-limitation "
                                       "statement in reports/13_old_protocol_archive/balanced_dynamic_dataset_handoff.md)")}
NA_ROWS = [("Dynamic->Dynamic", "N/A: no approved dynamic-source training protocol"),
           ("Dynamic->Static", "N/A: no approved dynamic-source training protocol")]
CLASS15_INDEX = 14  # class_folder_id 15 -> 0-based 14


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: Path, device: torch.device):
    """Build the model named by ``ck['model_config']['backbone']`` and load its weights.

    Backward-compatible dispatch (Step 6 / ViT extension, 2026-08-23): every existing config/checkpoint
    carries ``model.backbone: resnet50`` (see ``configs/source_only_resnet50.yaml``), so omitting the key
    defaults to the original, byte-for-byte-unchanged ``SourceClassifier`` path. Only ``model_config.backbone
    in {"vit_b_16", "vit_b16", "vit"}`` routes to the new ``ViTSourceClassifier`` (Step 6).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["model_config"]
    backbone = m.get("backbone", "resnet50")
    if backbone in ("vit_b_16", "vit_b16", "vit"):
        from src.tasks.icassp2027.model_vit import ViTSourceClassifier
        model = ViTSourceClassifier(num_classes=m["num_classes"], pretrained="none", proj_dim=m.get("proj_dim", 0))
    else:
        model = SourceClassifier(num_classes=m["num_classes"], pretrained="none", proj_dim=m.get("proj_dim", 0))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def run_model(model: SourceClassifier, loader: DataLoader, device: torch.device) -> dict:
    feats, logits, labels, ids, extra = [], [], [], [], {}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg, f = model(x, return_features=True)
        logits.append(lg.float().cpu()); feats.append(f.float().cpu())
        ids.extend(batch["sample_id"])
        if "label" in batch:
            labels.append(batch["label"])
        for k in ("block_id", "direction_id", "stream_position"):
            if k in batch:
                extra.setdefault(k, []).append(batch[k])
    out = {"features": torch.cat(feats).numpy(), "logits": torch.cat(logits).numpy(), "sample_id": np.array(ids)}
    if labels:
        out["label"] = torch.cat(labels).numpy()
    for k, v in extra.items():
        out[k] = torch.cat(v).numpy()
    return out


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, class_names: list[str]) -> dict:
    y_pred = logits.argmax(1)
    n_cls = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    n_per = cm.sum(1)
    per_class_acc = np.where(n_per > 0, np.diag(cm) / np.maximum(n_per, 1), np.nan)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_cls)), zero_division=0)
    out = {
        "n": int(len(y_true)),
        "top1": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_cls)), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_accuracy": {class_names[i]: (None if np.isnan(per_class_acc[i]) else float(per_class_acc[i])) for i in range(n_cls)},
        "samples_per_class": {class_names[i]: int(n_per[i]) for i in range(n_cls)},
        "confusion_matrix": cm.tolist(),
        "class15": {"name": class_names[CLASS15_INDEX], "n": int(n_per[CLASS15_INDEX]),
                    "precision": float(p[CLASS15_INDEX]), "recall": float(r[CLASS15_INDEX]), "f1": float(f[CLASS15_INDEX]),
                    "share_of_predictions": float((y_pred == CLASS15_INDEX).mean()),
                    "share_of_samples": float((y_true == CLASS15_INDEX).mean())},
    }
    out.update(prediction_stats(logits, class_names))
    return out


def prediction_stats(logits: np.ndarray, class_names: list[str]) -> dict:
    """Label-free prediction statistics: class histogram, entropy, collapse indicators."""
    z = logits - logits.max(1, keepdims=True); probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    hist = np.bincount(logits.argmax(1), minlength=len(class_names))
    return {"prediction_histogram": {class_names[i]: int(hist[i]) for i in range(len(class_names))},
            "prediction_entropy": {"mean": float(ent.mean()), "median": float(np.median(ent)), "p10": float(np.percentile(ent, 10)),
                                   "p90": float(np.percentile(ent, 90)), "max_possible": float(np.log(len(class_names)))},
            "prediction_confidence": {"mean": float(probs.max(1).mean()), "median": float(np.median(probs.max(1))),
                                      "share_above_0.9": float((probs.max(1) > 0.9).mean())},
            "collapse": {"max_class_share": float(hist.max() / hist.sum()), "argmax_class": class_names[int(hist.argmax())],
                         "n_classes_never_predicted": int((hist == 0).sum())}}


def evaluate_split(checkpoint: Path, split: str, batch_size: int, num_workers: int, device: torch.device,
                   limit: int | None = None) -> dict:
    if split not in LABELED_SPLITS:
        raise ValueError(split)
    model, ck = load_checkpoint(checkpoint, device)
    ds = LabeledManifestDataset(split, allowed_splits=SOURCE_SELECT_SPLITS + FINAL_EVAL_SPLITS, limit=limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, pin_memory=True)
    res = run_model(model, loader, device)
    metrics = compute_metrics(res["label"], res["logits"], ds.class_names)
    direction = DIRECTION[split]
    if ck.get("experiment") == "dynamic_oracle_resnet50" and split == "dynamic_test":
        direction = "Dynamic->Dynamic (supervised dynamic reference; diagnostic only, not TTA, not an upper bound)"
    elif ck.get("experiment") == "target_stats_resnet50" and split == "dynamic_test":
        direction = "Static->Dynamic after target-statistics (BN running-stats) adaptation; offline unsupervised, not strict online TTA"
    elif ck.get("experiment") == "tent_resnet50" and split == "dynamic_test":
        direction = "Static->Dynamic after Tent (entropy minimization, BN affine only); offline unsupervised, not strict online TTA"
    elif ck.get("experiment") == "proposed_resnet50" and split == "dynamic_test":
        direction = "Static->Dynamic after proposed entropy+consistency adaptation (BN affine only); offline unsupervised, not strict online TTA"
    metrics.update({"split": split, "direction": direction, "is_final_test": split in FINAL_EVAL_SPLITS,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                    "seed": ck.get("seed"), "experiment": ck.get("experiment"), "protocol": ck.get("protocol")})
    return metrics


def aggregate(exp_root: Path = Path("experiments/source_only"), report_dir: Path = Path("reports")) -> None:
    REPORT_CSV, REPORT_JSON = report_dir / "baseline_results.csv", report_dir / "baseline_results.json"
    rows = []
    for f in sorted(exp_root.glob("seed*/eval/eval_*.json")):
        rows.append(json.load(open(f)))
    if not rows:
        print("no eval_*.json found"); return
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "seed", "split", "direction", "is_final_test", "n", "top1", "macro_f1", "balanced_accuracy",
            "class15_recall", "class15_precision", "class15_f1", "class15_share_of_predictions", "checkpoint_sha256"]
    flat = []
    for r in rows:
        flat.append({"experiment": r["experiment"], "seed": r["seed"], "split": r["split"], "direction": r["direction"],
                     "is_final_test": r["is_final_test"], "n": r["n"], "top1": r["top1"], "macro_f1": r["macro_f1"],
                     "balanced_accuracy": r["balanced_accuracy"], "class15_recall": r["class15"]["recall"],
                     "class15_precision": r["class15"]["precision"], "class15_f1": r["class15"]["f1"],
                     "class15_share_of_predictions": r["class15"]["share_of_predictions"],
                     "checkpoint_sha256": r["checkpoint_sha256"][:16]})
    summary = []
    for split in LABELED_SPLITS:
        sel = [r for r in flat if r["split"] == split]
        if not sel:
            continue
        agg = {"experiment": sel[0]["experiment"], "seed": f"mean±std over {len(sel)} seeds", "split": split,
               "direction": sel[0]["direction"], "is_final_test": sel[0]["is_final_test"], "n": sel[0]["n"], "checkpoint_sha256": ""}
        for k in ("top1", "macro_f1", "balanced_accuracy", "class15_recall", "class15_precision", "class15_f1", "class15_share_of_predictions"):
            v = np.array([r[k] for r in sel]); agg[k] = f"{v.mean():.4f}±{v.std(ddof=0):.4f}"
        summary.append(agg)
    for direction, reason in NA_ROWS:
        summary.append({c: "" for c in cols} | {"experiment": flat[0]["experiment"], "seed": "-", "split": "-",
                                                "direction": direction, "is_final_test": "", "n": "", "top1": reason})
    with open(REPORT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(flat + summary)
    json.dump({"per_seed": rows, "summary": summary, "not_applicable": [dict(direction=d, reason=r) for d, r in NA_ROWS]},
              open(REPORT_JSON, "w"), indent=1)
    print(f"wrote {REPORT_CSV} ({len(flat)} rows + {len(summary)} summary/N-A rows) and {REPORT_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--split", choices=LABELED_SPLITS)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <checkpoint dir>/eval")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke only")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--exp-root", type=Path, default=Path("experiments/source_only"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.exp_root, a.report_dir); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = evaluate_split(a.checkpoint, a.split, a.batch_size, a.num_workers, device, a.limit)
    out_dir = a.out_dir or a.checkpoint.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(out_dir / f"eval_{a.split}.json", "w"), indent=1)
    print(f"{a.split} [{m['direction']}] n={m['n']} top1={m['top1']:.4f} macroF1={m['macro_f1']:.4f} "
          f"balAcc={m['balanced_accuracy']:.4f} class15 recall={m['class15']['recall']:.4f} "
          f"predShare={m['class15']['share_of_predictions']:.4f}")


if __name__ == "__main__":
    main()
