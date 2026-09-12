"""Step 5B — Tent-style entropy minimization (BatchNorm affine parameters only).

    python -m src.tasks.icassp2027.adapt_tent --config configs/tent_resnet50.yaml --seed 0 --variant natural --passes 1

Exact scope:
    * start from the SOURCE-ONLY checkpoint; conv/backbone weights and the classifier head are frozen
      (requires_grad=False) and asserted bit-identical afterwards;
    * trainable = gamma/beta of every BatchNorm2d only (list saved to trainable_parameters.json);
    * loss = mean softmax entropy over the batch (float32), SGD lr 1e-3, momentum 0.9, Nesterov, wd 0, no scheduler;
    * BN running statistics are RESET and re-estimated cumulatively (momentum=None) while each batch is normalized
      with its own statistics (Tent standard); the saved model is evaluated in eval mode (running statistics);
    * reads ONLY the label-free adaptation Parquet export, fixed stream order, batch 32; no labels, no pseudo-labels,
      no class balancing, no prototypes, no test data.  NaN/Inf in loss or parameters aborts the run.
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

from src.tasks.icassp2027.adapt_target_stats import bn_layers, snapshot_params
from src.tasks.icassp2027.data import AdaptationParquetDataset, assert_not_internal, collate
from src.tasks.icassp2027.evaluate import load_checkpoint, sha256_file
from src.tasks.icassp2027.train_source import env_info


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy -sum_c p log p, computed in float32."""
    logp = torch.log_softmax(logits.float(), dim=1)
    return -(logp.exp() * logp).sum(1)


def configure_tent(model: nn.Module) -> tuple[list[str], list[torch.nn.Parameter]]:
    """Freeze everything, then enable only BN gamma/beta. Returns (trainable names, params)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    names, params = [], []
    for mname, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None; m.train()   # batch statistics for normalization; cumulative running stats
            for pname, p in m.named_parameters(recurse=False):
                if pname in ("weight", "bias"):
                    p.requires_grad_(True); names.append(f"{mname}.{pname}"); params.append(p)
    if not params:
        raise RuntimeError("model has no BatchNorm affine parameters — Tent not applicable")
    return names, params


def run_tent(model: nn.Module, loader: DataLoader, passes: int, opt: torch.optim.Optimizer, device: torch.device, log,
             amp: bool = True, run=None, nan_guard: bool = True) -> list[dict]:
    records, step, n_cls = [], 0, model.head.out_features
    trainable = [p for p in model.parameters() if p.requires_grad]
    for ps in range(passes):
        t0 = time.time(); ents, losses = [], []; hist = np.zeros(n_cls, dtype=np.int64)
        for i, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(x)
            ent = softmax_entropy(logits); loss = ent.mean()
            if nan_guard and not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite entropy loss at pass {ps} batch {i}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            ents.append(ent.detach().mean().item()); losses.append(loss.detach().item()); hist += np.bincount(logits.detach().argmax(1).cpu().numpy(), minlength=n_cls)
            if nan_guard and any(not torch.isfinite(p).all() for p in trainable):
                raise FloatingPointError(f"non-finite BN affine parameter at pass {ps} batch {i}")
            if i % 50 == 0:
                log.write(json.dumps({"pass": ps, "batch": i, "step": step, "entropy": ents[-1], "loss": losses[-1],
                                      "stream_position_first": int(batch["stream_position"][0]),
                                      "max_abs_gamma": max(p.detach().abs().max().item() for n, p in model.named_parameters() if p.requires_grad and n.endswith("weight")),
                                      "max_abs_beta": max(p.detach().abs().max().item() for n, p in model.named_parameters() if p.requires_grad and n.endswith("bias"))}) + "\n")
                if run:
                    run.log({"adapt/entropy": ents[-1], "adapt/loss": losses[-1], "adapt/pass": ps, "adapt/batch": i}, step=step)
        rec = {"pass": ps, "batches": i + 1, "steps_total": step, "entropy_mean": float(np.mean(ents)), "entropy_first_batch": ents[0], "entropy_last_batch": ents[-1],
               "adapt_pred_histogram": hist.tolist(), "adapt_pred_max_share": float(hist.max() / hist.sum()), "adapt_pred_n_never": int((hist == 0).sum()),
               "pass_time_s": time.time() - t0, "gpu_mem_peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None}
        records.append(rec); log.write(json.dumps({"pass_done": rec}) + "\n"); log.flush()
        if run:
            run.log({"adapt/pass_entropy_mean": rec["entropy_mean"], "adapt/pass_pred_max_share": rec["adapt_pred_max_share"], "adapt/pass_time_s": rec["pass_time_s"],
                     "adapt/gpu_mem_peak_gb": rec["gpu_mem_peak_gb"], "adapt/pass_idx": ps}, step=step)
    model.eval()
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
    ap.add_argument("--n-target", type=int, default=None,
                    help="deterministic stream_position-order prefix count for the adaptation-data-amount "
                         "ablation (2026-08-23 amendment); unlike --limit this does NOT set smoke=True")
    ap.add_argument("--wandb-group", type=str, default=None, help="override the config's wandb.group")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); A = cfg["adaptation"]; V = A["variants"][a.variant]
    assert a.passes in {A["passes_primary"], *A["passes_secondary"]}, "pass count must be pre-specified"
    src_ck = Path(cfg["source_checkpoints"][a.seed]); assert_not_internal(src_ck)
    assert "source_only" in str(src_ck), "Tent must start from a source-only checkpoint"
    out = a.out or Path(cfg["output_root"]) / f"seed{a.seed}" / a.variant / f"pass{a.passes}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(a.seed)

    src_sha = sha256_file(src_ck)
    model, ck = load_checkpoint(src_ck, device)
    params_before = snapshot_params(model)
    names, params = configure_tent(model)
    opt = torch.optim.SGD(params, lr=A["lr"], momentum=A["momentum"], nesterov=A["nesterov"], weight_decay=A["weight_decay"])
    effective_limit = a.limit if a.limit is not None else a.n_target
    ds = AdaptationParquetDataset(a.variant, export_dir=Path(V["export"]).parent, limit=effective_limit)
    assert a.limit or a.n_target or len(ds) == A["n_target"], len(ds)
    loader = DataLoader(ds, batch_size=A["batch_size"], shuffle=False, num_workers=a.num_workers, collate_fn=collate, pin_memory=True)
    env = {k: str(v) for k, v in env_info().items()}
    resolved = {"experiment": cfg["experiment"], "method": cfg["method"], "seed": a.seed, "variant": a.variant, "variant_role": V["role"],
                "adaptation_export": V["export"], "adaptation_manifest": V["manifest"], "adaptation_protocol_name": ds.metadata.get("adaptation_protocol"),
                "stream_id": f"{ds.split}@stream_position (seed {ds.metadata.get('selection_seed')})", "n_target": len(ds), "batch_size": A["batch_size"], "passes": a.passes,
                "update_scope": "bn_affine_only", "n_trainable_tensors": len(params), "n_trainable_scalars": int(sum(p.numel() for p in params)),
                "n_frozen_scalars": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)), "frozen": A["frozen"],
                "bn_running_stats": A["bn_running_stats"], "eval_bn_mode": A["eval_bn_mode"], "loss": A["loss"],
                "optimizer": {"type": "sgd", "lr": A["lr"], "momentum": A["momentum"], "nesterov": A["nesterov"], "weight_decay": A["weight_decay"], "scheduler": "none"},
                "amp": A["amp"], "source_checkpoint": str(src_ck), "source_checkpoint_sha256": src_sha, "source_experiment": ck.get("experiment"),
                "split_id": "icassp2027_spad_electronics_v1", "adaptation_export_columns": ds.columns, "env": env, "smoke": bool(a.limit),
                "data_amount_ablation": ({"requested_n_target": a.n_target, "prefix_rule": "stream_position ascending order, deterministic first-N slice, no class information"} if a.n_target else None),
                "command": f"python -m src.tasks.icassp2027.adapt_tent --config {a.config} --seed {a.seed} --variant {a.variant} --passes {a.passes}" + (f" --n-target {a.n_target}" if a.n_target else "")}
    yaml.safe_dump(resolved, open(out / "config.yaml", "w"), sort_keys=False); json.dump(env, open(out / "env.json", "w"), indent=1)
    json.dump({"trainable": names, "n_trainable_scalars": resolved["n_trainable_scalars"]}, open(out / "trainable_parameters.json", "w"), indent=1)

    run = None
    if cfg.get("wandb", {}).get("enabled") and not a.limit:
        try:
            import wandb
            w = cfg["wandb"]
            run = wandb.init(entity=w.get("entity"), project=w["project"], group=(a.wandb_group or w["group"]), job_type="adapt", tags=w["tags"] + [f"seed{a.seed}", a.variant, f"pass{a.passes}"],
                             name=f"tent_{a.variant}_pass{a.passes}_seed{a.seed}", config=resolved, mode=w.get("mode", "online"), dir=str(out))
            (out / "wandb_run.json").write_text(json.dumps({"id": run.id, "project": run.project, "entity": run.entity, "url": run.url}))
        except Exception as e:
            (out / "wandb_run.json").write_text(json.dumps({"id": None, "error": str(e)})); run = None

    log = open(out / "adapt_log.jsonl", "w")
    status = "ok"
    try:
        records = run_tent(model, loader, a.passes, opt, device, log, amp=A["amp"] == "bf16", run=run, nan_guard=A["nan_guard"] == "abort")
    except FloatingPointError as e:
        status = f"aborted: {e}"; records = []; log.write(json.dumps({"abort": str(e)}) + "\n")
    log.close()

    # ---- parameter-difference report
    params_after = snapshot_params(model); diff = {}
    for n in params_before:
        d = (params_after[n].float() - params_before[n].float())
        diff[n] = {"trainable": n in names, "changed": bool(not torch.equal(params_before[n], params_after[n])), "mean_abs_change": float(d.abs().mean()),
                   "max_abs_change": float(d.abs().max()), "max_abs_after": float(params_after[n].abs().max())}
    frozen_changed = [n for n, v in diff.items() if not v["trainable"] and v["changed"]]
    assert not frozen_changed, f"frozen parameters changed: {frozen_changed[:5]}"
    assert all(p.grad is None for n, p in model.named_parameters() if n not in names), "gradient present on a frozen parameter"
    gam = [v for n, v in diff.items() if v["trainable"] and n.endswith("weight")]; bet = [v for n, v in diff.items() if v["trainable"] and n.endswith("bias")]
    summary = {"status": status, "frozen_unchanged": True, "n_trainable_changed": sum(v["changed"] for v in diff.values() if v["trainable"]),
               "gamma_mean_abs_change": float(np.mean([v["mean_abs_change"] for v in gam])), "gamma_max_abs_change": float(max(v["max_abs_change"] for v in gam)),
               "gamma_max_abs_after": float(max(v["max_abs_after"] for v in gam)), "beta_mean_abs_change": float(np.mean([v["mean_abs_change"] for v in bet])),
               "beta_max_abs_change": float(max(v["max_abs_change"] for v in bet)), "beta_max_abs_after": float(max(v["max_abs_after"] for v in bet)),
               "passes": records, "source_checkpoint_sha256": src_sha}
    json.dump(diff, open(out / "parameter_diff.json", "w"), indent=1)
    ck_out = {"state_dict": model.state_dict(), "model_config": ck["model_config"], "seed": a.seed, "experiment": cfg["experiment"], "protocol": cfg["protocol"],
              "adaptation": resolved, "source_checkpoint_sha256": src_sha, "passes": records, "status": status}
    torch.save(ck_out, out / "adapted_model.pt"); sha = sha256_file(out / "adapted_model.pt")
    summary["adapted_checkpoint_sha256"] = sha
    (out / "checkpoint.sha256").write_text(f"{sha}  adapted_model.pt\n{src_sha}  source:{src_ck}\n")
    json.dump(summary, open(out / "adapt_metrics.json", "w"), indent=1)
    if run:
        run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}); run.finish()
    print(f"seed{a.seed} {a.variant} pass{a.passes}: {status}; {len(ds)} samples, {records[-1]['steps_total'] if records else 0} steps; "
          f"entropy {records[0]['entropy_first_batch']:.3f}->{records[-1]['entropy_last_batch']:.3f}; gamma max|d| {summary['gamma_max_abs_change']:.4f}; "
          f"frozen unchanged; sha256 {sha[:16]}; out {out}" if records else f"{status}; out {out}")
    if status != "ok":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
