"""Step 5C — proposed method: entropy minimization + augmentation consistency (BN affine only).

    python -m src.tasks.icassp2027.adapt_proposed --config configs/proposed_resnet50.yaml --seed 0 --variant natural --passes 1
    python -m src.tasks.icassp2027.adapt_proposed ... --lambda-ent 0 --lambda-cons 1 --ablation consistency_only

Per batch x (fixed stream, batch 32):
    x_w = x (identity);  x_s = renorm(clamp(denorm(x) + N(0, sigma^2), 0, 1))   — noise in [0,1] image space
    p_w = softmax(f(x_w));  p_s = softmax(f(x_s))
    L = lambda_ent * H(p_w) + lambda_cons * KL(stopgrad(p_w) || p_s)      (float32)
Trainable: BN gamma/beta only (same 106 tensors as Tent); conv weights + head frozen and asserted bit-identical.
BN handling (P1): weak forward = Tent forward (batch statistics; running stats re-estimated cumulatively); the strong
forward uses its own batch statistics but running-stat buffers are restored afterwards, so running statistics come from
clean views only.  Reads only the label-free adaptation export; never dynamic_test or configs/manifests/internal/.
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
from src.tasks.icassp2027.adapt_tent import configure_tent, softmax_entropy
from src.tasks.icassp2027.data import IMAGENET_MEAN, IMAGENET_STD, AdaptationParquetDataset, assert_not_internal, collate
from src.tasks.icassp2027.evaluate import load_checkpoint, sha256_file
from src.tasks.icassp2027.train_source import env_info


def gaussian_noise_image_space(x: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """x: normalized (B,3,H,W) with 3 identical channels. Noise is added in [0,1] image space and re-normalized."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x01 = x * std + mean                                                    # exact inverse normalization
    noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype, generator=gen) * sigma
    x01 = (x01 + noise).clamp_(0.0, 1.0)                                    # one channel of noise, replicated -> L stays [L,L,L]
    return (x01 - mean) / std


def kl_consistency(p_w_logits: torch.Tensor, p_s_logits: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(stopgrad(p_w) || p_s) in float32."""
    logp_w = torch.log_softmax(p_w_logits.float(), 1).detach()
    logp_s = torch.log_softmax(p_s_logits.float(), 1)
    return (logp_w.exp() * (logp_w - logp_s)).sum(1)


class BNStatGuard:
    """Forward pass that must not contribute to BN running statistics (P1).

    Inside the guard every BatchNorm keeps train mode (normalizes with the current batch's own statistics) but
    ``track_running_stats`` is switched off, so no running_mean/var/num_batches_tracked buffer is written.
    (Restoring buffers by in-place copy instead would bump their autograd version counters and break the backward
    of the preceding weak forward.)
    """
    def __init__(self, model: nn.Module):
        self.bns = bn_layers(model)
    def __enter__(self):
        for b in self.bns:
            b.track_running_stats = False
    def __exit__(self, *exc):
        for b in self.bns:
            b.track_running_stats = True


def _bn_buffers(bns):
    return [(b.running_mean.detach().clone(), b.running_var.detach().clone(), b.num_batches_tracked.detach().clone()) for b in bns]


def run_proposed(model, loader, passes, opt, device, log, sigma, lam_ent, lam_cons, seed, amp=True, run=None, nan_guard=True):
    """Safeguards (approved 2026-08-23): a loss term is computed and added ONLY if its lambda != 0 (an unused KL branch can
    never contaminate the loss); every strong-view forward is asserted not to modify any BN running buffer; each term
    and the total are checked finite; p_w/p_s are softmax probabilities and stop-gradient is applied to p_w only."""
    records, step, n_cls = [], 0, model.head.out_features
    gen = torch.Generator(device=device.type).manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]; bns = bn_layers(model)
    use_ent, use_cons = lam_ent != 0, lam_cons != 0
    for ps in range(passes):
        t0 = time.time(); E, C, T, CONF = [], [], [], []; hist = np.zeros(n_cls, dtype=np.int64); bn_checks = 0
        for i, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits_w = model(x)                                         # weak = identity; updates running stats (Tent forward)
                logits_s = None
                if use_cons:
                    before = _bn_buffers(bns)
                    with BNStatGuard(model):
                        logits_s = model(gaussian_noise_image_space(x, sigma, gen))   # strong view: batch stats, NO buffer writes
                    for b, (m, v, n) in zip(bns, before):
                        same = lambda p, q: torch.allclose(p, q, rtol=0.0, atol=0.0, equal_nan=True)   # exact equality, NaN-aware
                        if not (same(b.running_mean, m) and same(b.running_var, v) and torch.equal(b.num_batches_tracked, n)):
                            raise RuntimeError(f"strong-view forward modified BN running statistics at pass {ps} batch {i}")
                    bn_checks += 1
            ent = softmax_entropy(logits_w)                                 # H(p_w), p_w = softmax(logits_w), float32
            cons = kl_consistency(logits_w, logits_s) if use_cons else None   # KL(stopgrad(p_w) || p_s)
            if nan_guard and (not torch.isfinite(ent).all() or (cons is not None and not torch.isfinite(cons).all())):
                raise FloatingPointError(f"non-finite loss term at pass {ps} batch {i}")
            loss = torch.zeros((), device=device, dtype=torch.float32)
            if use_ent:
                loss = loss + lam_ent * ent.mean()
            if use_cons:
                loss = loss + lam_cons * cons.mean()
            if nan_guard and not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at pass {ps} batch {i}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            E.append(ent.detach().mean().item()); C.append(cons.detach().mean().item() if cons is not None else None); T.append(loss.detach().item())
            CONF.append(torch.softmax(logits_w.detach().float(), 1).max(1).values.mean().item())
            hist += np.bincount(logits_w.detach().argmax(1).cpu().numpy(), minlength=n_cls)
            if nan_guard and any(not torch.isfinite(p).all() for p in trainable):
                raise FloatingPointError(f"non-finite BN affine parameter at pass {ps} batch {i}")
            if i % 50 == 0:
                rec = {"pass": ps, "batch": i, "step": step, "entropy": E[-1], "consistency": C[-1], "total": T[-1], "confidence_w": CONF[-1],
                       "stream_position_first": int(batch["stream_position"][0]),
                       "max_abs_gamma": max(p.detach().abs().max().item() for n, p in model.named_parameters() if p.requires_grad and n.endswith("weight")),
                       "max_abs_beta": max(p.detach().abs().max().item() for n, p in model.named_parameters() if p.requires_grad and n.endswith("bias"))}
                log.write(json.dumps(rec) + "\n")
                if run:
                    run.log({k: v for k, v in {"adapt/entropy": E[-1], "adapt/consistency": C[-1], "adapt/total": T[-1], "adapt/confidence_w": CONF[-1], "adapt/pass": ps, "adapt/batch": i}.items() if v is not None}, step=step)
        Cv = [c for c in C if c is not None]
        rec = {"pass": ps, "batches": i + 1, "steps_total": step, "entropy_mean": float(np.mean(E)), "consistency_mean": (float(np.mean(Cv)) if Cv else None), "total_mean": float(np.mean(T)),
               "confidence_w_mean": float(np.mean(CONF)), "entropy_first_batch": E[0], "entropy_last_batch": E[-1], "consistency_first_batch": C[0], "consistency_last_batch": C[-1],
               "lambda_ent": lam_ent, "lambda_cons": lam_cons, "strong_view_bn_buffer_checks": bn_checks, "strong_view_bn_untouched": True,
               "adapt_pred_histogram": hist.tolist(), "adapt_pred_max_share": float(hist.max() / hist.sum()), "adapt_pred_n_never": int((hist == 0).sum()),
               "pass_time_s": time.time() - t0, "gpu_mem_peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None}
        records.append(rec); log.write(json.dumps({"pass_done": rec}) + "\n"); log.flush()
        if run:
            run.log({"adapt/pass_entropy_mean": rec["entropy_mean"], "adapt/pass_consistency_mean": rec["consistency_mean"] if rec["consistency_mean"] is not None else float("nan"), "adapt/pass_total_mean": rec["total_mean"],
                     "adapt/pass_confidence_w_mean": rec["confidence_w_mean"], "adapt/pass_pred_max_share": rec["adapt_pred_max_share"], "adapt/pass_time_s": rec["pass_time_s"],
                     "adapt/gpu_mem_peak_gb": rec["gpu_mem_peak_gb"], "adapt/pass_idx": ps}, step=step)
    model.eval()
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["natural", "balanced"], required=True)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--lambda-ent", type=float, default=None, help="override (ablation); default from config")
    ap.add_argument("--lambda-cons", type=float, default=None)
    ap.add_argument("--ablation", default="proposed", help="output sub-directory name: proposed | consistency_only")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="smoke only")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-target", type=int, default=None,
                    help="deterministic stream_position-order prefix count for the adaptation-data-amount "
                         "ablation (2026-08-23 amendment); unlike --limit this does NOT set smoke=True")
    ap.add_argument("--wandb-group", type=str, default=None, help="override the config's wandb.group")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); A = cfg["adaptation"]; V = A["variants"][a.variant]
    lam_ent = A["loss"]["lambda_ent"] if a.lambda_ent is None else a.lambda_ent
    lam_cons = A["loss"]["lambda_cons"] if a.lambda_cons is None else a.lambda_cons
    assert a.passes in {A["passes_primary"], *A["passes_secondary"]}
    src_ck = Path(cfg["source_checkpoints"][a.seed]); assert_not_internal(src_ck); assert "source_only" in str(src_ck)
    out = a.out or Path(cfg["output_root"]) / a.ablation / f"seed{a.seed}" / a.variant / f"pass{a.passes}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(a.seed)

    src_sha = sha256_file(src_ck); model, ck = load_checkpoint(src_ck, device)
    params_before = snapshot_params(model)
    names, params = configure_tent(model)                                   # same trainable set + BN setup as Tent
    opt = torch.optim.SGD(params, lr=A["lr"], momentum=A["momentum"], nesterov=A["nesterov"], weight_decay=A["weight_decay"])
    effective_limit = a.limit if a.limit is not None else a.n_target
    ds = AdaptationParquetDataset(a.variant, export_dir=Path(V["export"]).parent, limit=effective_limit); assert a.limit or a.n_target or len(ds) == A["n_target"]
    loader = DataLoader(ds, batch_size=A["batch_size"], shuffle=False, num_workers=a.num_workers, collate_fn=collate, pin_memory=True)
    env = {k: str(v) for k, v in env_info().items()}
    resolved = {"experiment": cfg["experiment"], "method": cfg["method"], "ablation": a.ablation, "seed": a.seed, "variant": a.variant, "variant_role": V["role"],
                "adaptation_export": V["export"], "adaptation_manifest": V["manifest"], "adaptation_protocol_name": ds.metadata.get("adaptation_protocol"),
                "stream_id": f"{ds.split}@stream_position (seed {ds.metadata.get('selection_seed')})", "n_target": len(ds), "batch_size": A["batch_size"], "passes": a.passes,
                "update_scope": "bn_affine_only", "n_trainable_tensors": len(params), "n_trainable_scalars": int(sum(p.numel() for p in params)),
                "n_frozen_scalars": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)), "frozen": A["frozen"],
                "bn_running_stats": A["bn_running_stats"], "eval_bn_mode": A["eval_bn_mode"], "weak_augmentation": A["weak_augmentation"],
                "strong_augmentation": A["strong_augmentation"], "noise_sigma": A["noise_sigma"], "noise_space": A["noise_space"], "noise_seed": a.seed,
                "loss": {"entropy": "H(p_w)", "consistency": "KL(stopgrad(p_w)||p_s)", "lambda_ent": lam_ent, "lambda_cons": lam_cons},
                "optimizer": {"type": "sgd", "lr": A["lr"], "momentum": A["momentum"], "nesterov": A["nesterov"], "weight_decay": A["weight_decay"], "scheduler": "none"},
                "amp": A["amp"], "source_checkpoint": str(src_ck), "source_checkpoint_sha256": src_sha, "source_experiment": ck.get("experiment"),
                "split_id": "icassp2027_spad_electronics_v1", "adaptation_export_columns": ds.columns, "env": env, "smoke": bool(a.limit),
                "data_amount_ablation": ({"requested_n_target": a.n_target, "prefix_rule": "stream_position ascending order, deterministic first-N slice, no class information"} if a.n_target else None),
                "command": f"python -m src.tasks.icassp2027.adapt_proposed --config {a.config} --seed {a.seed} --variant {a.variant} --passes {a.passes} --lambda-ent {lam_ent} --lambda-cons {lam_cons} --ablation {a.ablation}" + (f" --n-target {a.n_target}" if a.n_target else "")}
    yaml.safe_dump(resolved, open(out / "config.yaml", "w"), sort_keys=False); json.dump(env, open(out / "env.json", "w"), indent=1)
    json.dump({"trainable": names, "n_trainable_scalars": resolved["n_trainable_scalars"]}, open(out / "trainable_parameters.json", "w"), indent=1)

    run = None
    if cfg.get("wandb", {}).get("enabled") and not a.limit:
        try:
            import wandb
            w = cfg["wandb"]
            run = wandb.init(entity=w.get("entity"), project=w["project"], group=(a.wandb_group or w["group"]), job_type="adapt",
                             tags=w["tags"] + [f"seed{a.seed}", a.variant, f"pass{a.passes}", a.ablation],
                             name=f"{a.ablation}_{a.variant}_pass{a.passes}_seed{a.seed}", config=resolved, mode=w.get("mode", "online"), dir=str(out))
            (out / "wandb_run.json").write_text(json.dumps({"id": run.id, "project": run.project, "entity": run.entity, "url": run.url}))
        except Exception as e:
            (out / "wandb_run.json").write_text(json.dumps({"id": None, "error": str(e)})); run = None

    log = open(out / "adapt_log.jsonl", "w"); status = "ok"
    try:
        records = run_proposed(model, loader, a.passes, opt, device, log, A["noise_sigma"], lam_ent, lam_cons, a.seed, amp=A["amp"] == "bf16", run=run, nan_guard=A["nan_guard"] == "abort")
    except FloatingPointError as e:
        status = f"aborted: {e}"; records = []; log.write(json.dumps({"abort": str(e)}) + "\n")
    log.close()

    params_after = snapshot_params(model); diff = {}
    for n in params_before:
        d = params_after[n].float() - params_before[n].float()
        diff[n] = {"trainable": n in names, "changed": bool(not torch.equal(params_before[n], params_after[n])), "mean_abs_change": float(d.abs().mean()),
                   "max_abs_change": float(d.abs().max()), "max_abs_after": float(params_after[n].abs().max())}
    assert not [n for n, v in diff.items() if not v["trainable"] and v["changed"]], "frozen parameters changed"
    assert all(p.grad is None for n, p in model.named_parameters() if n not in names), "gradient on a frozen parameter"
    gam = [v for n, v in diff.items() if v["trainable"] and n.endswith("weight")]; bet = [v for n, v in diff.items() if v["trainable"] and n.endswith("bias")]
    summary = {"status": status, "frozen_unchanged": True, "n_trainable_changed": sum(v["changed"] for v in diff.values() if v["trainable"]),
               "gamma_mean_abs_change": float(np.mean([v["mean_abs_change"] for v in gam])), "gamma_max_abs_change": float(max(v["max_abs_change"] for v in gam)),
               "gamma_max_abs_after": float(max(v["max_abs_after"] for v in gam)), "beta_mean_abs_change": float(np.mean([v["mean_abs_change"] for v in bet])),
               "beta_max_abs_change": float(max(v["max_abs_change"] for v in bet)), "beta_max_abs_after": float(max(v["max_abs_after"] for v in bet)),
               "passes": records, "source_checkpoint_sha256": src_sha, "lambda_ent": lam_ent, "lambda_cons": lam_cons}
    json.dump(diff, open(out / "parameter_diff.json", "w"), indent=1)
    ck_out = {"state_dict": model.state_dict(), "model_config": ck["model_config"], "seed": a.seed, "experiment": cfg["experiment"], "protocol": cfg["protocol"],
              "adaptation": resolved, "source_checkpoint_sha256": src_sha, "passes": records, "status": status}
    torch.save(ck_out, out / "adapted_model.pt"); sha = sha256_file(out / "adapted_model.pt"); summary["adapted_checkpoint_sha256"] = sha
    (out / "checkpoint.sha256").write_text(f"{sha}  adapted_model.pt\n{src_sha}  source:{src_ck}\n")
    json.dump(summary, open(out / "adapt_metrics.json", "w"), indent=1)
    if run:
        run.summary.update({k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))}); run.finish()
    if records:
        c0, c1 = records[0]["consistency_first_batch"], records[-1]["consistency_last_batch"]
        print(f"seed{a.seed} {a.variant} pass{a.passes} [{a.ablation}]: {status}; {len(ds)} samples, {records[-1]['steps_total']} steps; entropy {records[0]['entropy_first_batch']:.3f}->{records[-1]['entropy_last_batch']:.3f}; "
              f"consistency {'n/a' if c0 is None else f'{c0:.4f}->{c1:.4f}'}; strong-view BN checks {sum(r['strong_view_bn_buffer_checks'] for r in records)}; gamma max|d| {summary['gamma_max_abs_change']:.4f}; frozen unchanged; sha256 {sha[:16]}; out {out}")
    else:
        print(f"{status}; out {out}")
    if status != "ok":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
