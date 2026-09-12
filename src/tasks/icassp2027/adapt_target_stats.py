"""Step 5A — target-statistics adaptation (BatchNorm running statistics only).

    python -m src.tasks.icassp2027.adapt_target_stats --config configs/target_stats_resnet50.yaml --seed 0 --variant natural --passes 1

What it does, exactly:
    * loads a source-only checkpoint; every parameter (conv/linear weights, classifier, BN gamma/beta) is frozen
      (requires_grad=False) and never modified — asserted bit-for-bit after adaptation;
    * every BatchNorm2d has its running_mean/running_var/num_batches_tracked RESET and momentum=None, so after
      streaming the unlabeled adaptation export once the buffers hold the cumulative target mean/variance;
    * forward passes only (torch.no_grad), fixed stream order (stream_position), batch 32, no future batches;
    * reads ONLY the adaptation Parquet export (embedded bytes; no label / path / frame_index columns);
      never dynamic_test, never configs/manifests/internal/.
Evaluation on dynamic_test is done afterwards by src.tasks.icassp2027.evaluate (post hoc).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.tasks.icassp2027.data import AdaptationParquetDataset, assert_not_internal, collate
from src.tasks.icassp2027.evaluate import load_checkpoint, sha256_file
from src.tasks.icassp2027.train_source import env_info


def bn_layers(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def snapshot_bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: b.detach().clone() for n, b in model.named_buffers() if n.endswith(("running_mean", "running_var"))}


@torch.no_grad()
def adapt_bn_statistics(model: nn.Module, loader: DataLoader, passes: int, device: torch.device, log, run=None) -> list[dict]:
    """Normalization-only adaptation. Returns per-pass records."""
    bns = bn_layers(model)
    if not bns:
        raise RuntimeError("model exposes no BatchNorm layers with running statistics — target-statistics adaptation not applicable")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()                      # everything in inference mode ...
    for bn in bns:                    # ... except BN statistics, which we re-estimate
        bn.reset_running_stats(); bn.momentum = None; bn.train()
    records = []; step = 0
    for ps in range(passes):
        t0 = time.time()
        for i, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            model(x)                  # forward only; updates running_mean/var via cumulative average
            step += 1
            if i % 100 == 0:
                log.write(json.dumps({"pass": ps, "batch": i, "step": step, "stream_position_first": int(batch["stream_position"][0]),
                                      "bn0_running_mean_abs": float(bns[0].running_mean.abs().mean()), "bn0_running_var_mean": float(bns[0].running_var.mean())}) + "\n")
        rec = {"pass": ps, "batches": i + 1, "steps_total": step, "num_batches_tracked_bn0": int(bns[0].num_batches_tracked), "pass_time_s": time.time() - t0,
               "gpu_mem_peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None}
        records.append(rec); log.write(json.dumps({"pass_done": rec}) + "\n"); log.flush()
        if run:
            run.log({"adapt/pass": ps, "adapt/batches": rec["batches"], "adapt/pass_time_s": rec["pass_time_s"], "adapt/gpu_mem_peak_gb": rec["gpu_mem_peak_gb"]}, step=ps)
    for bn in bns:
        bn.eval()
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["natural", "balanced"], required=True)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="smoke only")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); A = cfg["adaptation"]; V = A["variants"][a.variant]
    assert a.passes in {A["passes_primary"], *A["passes_secondary"]}, "pass count must be pre-specified"
    src_ck = Path(cfg["source_checkpoints"][a.seed]); assert_not_internal(src_ck)
    out = a.out or Path(cfg["output_root"]) / f"seed{a.seed}" / a.variant / f"pass{a.passes}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    src_sha = sha256_file(src_ck)
    model, ck = load_checkpoint(src_ck, device)
    params_before = snapshot_params(model); bn_before = snapshot_bn_buffers(model)
    ds = AdaptationParquetDataset(a.variant, export_dir=Path(V["export"]).parent, limit=a.limit)   # label-free export only
    assert not a.limit and len(ds) == A["n_target"] or a.limit, len(ds)
    loader = DataLoader(ds, batch_size=A["batch_size"], shuffle=False, num_workers=a.num_workers, collate_fn=collate, pin_memory=True)
    env = {k: str(v) for k, v in env_info().items()}
    resolved = {"experiment": cfg["experiment"], "method": cfg["method"], "seed": a.seed, "variant": a.variant, "variant_role": V["role"],
                "adaptation_export": V["export"], "adaptation_manifest": V["manifest"], "adaptation_protocol_name": ds.metadata.get("adaptation_protocol"),
                "stream_id": f"{ds.split}@stream_position (seed {ds.metadata.get('selection_seed')})", "n_target": len(ds), "batch_size": A["batch_size"],
                "passes": a.passes, "bn_momentum": None, "reset_running_stats": True, "update_scope": "bn_running_stats_only",
                "frozen": A["frozen"], "gradients": "none", "source_checkpoint": str(src_ck), "source_checkpoint_sha256": src_sha,
                "source_experiment": ck.get("experiment"), "split_id": "icassp2027_spad_electronics_v1", "n_bn_layers": len(bn_layers(model)),
                "adaptation_export_columns": ds.columns, "env": env, "smoke": bool(a.limit),
                "command": f"python -m src.tasks.icassp2027.adapt_target_stats --config {a.config} --seed {a.seed} --variant {a.variant} --passes {a.passes}"}
    yaml.safe_dump(resolved, open(out / "config.yaml", "w"), sort_keys=False); json.dump(env, open(out / "env.json", "w"), indent=1)

    run = None
    if cfg.get("wandb", {}).get("enabled") and not a.limit:
        try:
            import wandb
            w = cfg["wandb"]
            run = wandb.init(entity=w.get("entity"), project=w["project"], group=w["group"], job_type="adapt",
                             tags=w["tags"] + [f"seed{a.seed}", a.variant, f"pass{a.passes}"],
                             name=f"target_stats_{a.variant}_pass{a.passes}_seed{a.seed}", config=resolved, mode=w.get("mode", "online"), dir=str(out))
            (out / "wandb_run.json").write_text(json.dumps({"id": run.id, "project": run.project, "entity": run.entity, "url": run.url}))
        except Exception as e:
            (out / "wandb_run.json").write_text(json.dumps({"id": None, "error": str(e)})); run = None

    log = open(out / "adapt_log.jsonl", "w")
    records = adapt_bn_statistics(model, loader, a.passes, device, log, run); log.close()

    # ---- integrity: parameters untouched, only BN buffers changed
    params_after = snapshot_params(model)
    assert all(torch.equal(params_before[n], params_after[n]) for n in params_before), "parameters changed — protocol violation"
    bn_after = snapshot_bn_buffers(model)
    drift = {n: float((bn_after[n] - bn_before[n]).abs().mean()) for n in bn_before}
    ck_out = {"state_dict": model.state_dict(), "model_config": ck["model_config"], "seed": a.seed, "experiment": cfg["experiment"],
              "protocol": cfg["protocol"], "adaptation": resolved, "source_checkpoint_sha256": src_sha, "passes": records}
    torch.save(ck_out, out / "adapted_model.pt"); sha = sha256_file(out / "adapted_model.pt")
    (out / "checkpoint.sha256").write_text(f"{sha}  adapted_model.pt\n{src_sha}  source:{src_ck}\n")
    json.dump({"passes": records, "bn_buffer_mean_abs_change": drift, "mean_running_mean_abs_change": float(np.mean([v for n, v in drift.items() if n.endswith("running_mean")])),
               "mean_running_var_abs_change": float(np.mean([v for n, v in drift.items() if n.endswith("running_var")])), "params_unchanged": True,
               "adapted_checkpoint_sha256": sha, "source_checkpoint_sha256": src_sha}, open(out / "adapt_metrics.json", "w"), indent=1)
    if run:
        run.summary.update({"adapted_checkpoint_sha256": sha, "source_checkpoint_sha256": src_sha, "params_unchanged": True,
                            "bn_running_mean_abs_change": float(np.mean([v for n, v in drift.items() if n.endswith("running_mean")]))}); run.finish()
    print(f"seed{a.seed} {a.variant} pass{a.passes}: {len(ds)} samples, {records[-1]['steps_total']} forward batches, params unchanged, sha256 {sha[:16]}, out {out}")


if __name__ == "__main__":
    main()
