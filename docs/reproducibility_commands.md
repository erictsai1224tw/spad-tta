# Reproducibility commands

Adapted from the source monorepo's internal reproducibility log, trimmed to the six methods shipped
in this repository and to commands that make sense against the sample data included here. All
commands assume you are at the repo root with dependencies from `requirements.txt` installed.

## 1. Environment

Recorded environment for the full-scale runs this protocol was validated against: Python 3.12,
torch 2.9.1, torchvision 0.24.1, CUDA 12.8. Any reasonably recent torch/torchvision (see
`requirements.txt`) should work; only the split/stream-order seeds in `configs/splits.yaml` are
guaranteed to reproduce exactly (see §6, the determinism caveat).

## 2. Source-only training + evaluation

```bash
# --smoke: 64 samples, 2 epochs, 2 steps/epoch — runs against the sample data in this repo
python -m src.tasks.icassp2027.train_source --config configs/source_only_resnet50.yaml --seed 0 --smoke

# Full-scale (needs the full dataset, not the sample shipped here):
for SEED in 0 1 2; do
  python -m src.tasks.icassp2027.train_source --config configs/source_only_resnet50.yaml --seed $SEED
done
# -> experiments/source_only/seed<k>/{best_model.pt,last_model.pt,config.yaml,train_log.jsonl,val_metrics.json,env.json,checkpoint.sha256}

python -m src.tasks.icassp2027.evaluate --checkpoint experiments/source_only/seed0/best_model.pt --split dynamic_test --limit 32
```

## 3. Dynamic supervised reference (`train_dynamic_oracle.py`)

```bash
python -m src.tasks.icassp2027.train_dynamic_oracle --config configs/dynamic_oracle_resnet50.yaml --seed 0 --smoke
```

**Needs the full dataset, not runnable against the sample export shipped here**: this script builds
its training pool by walking every file in `data/SPAD_Electronics/100_HZ/<class>/` for blocks 0–7
(24,355 frames) directly off disk and raises `FileNotFoundError` on any missing file, before
`--smoke` gets a chance to subsample — the full dynamic pool must be present.

## 4. LODO direction pilot (`train_lodo_direction.py`, `evaluate_lodo_direction.py`)

```bash
python -m src.tasks.icassp2027.train_lodo_direction --config configs/lodo_direction/leave_out_1.yaml --smoke --no-wandb
# -> experiments/lodo_direction/leave_out_1/{final_model.pt,config.yaml,train_log.jsonl,train_metrics.json,env.json,checkpoint.sha256}

python -m src.tasks.icassp2027.evaluate_lodo_direction --run experiments/lodo_direction/leave_out_1 --limit 32 --no-wandb
```

Repeat for `leave_out_{2,3,4,5}.yaml` for the other four folds.

## 5. BN target-statistics adaptation (`adapt_target_stats.py`)

```bash
python -m src.tasks.icassp2027.adapt_target_stats --config configs/target_stats_resnet50.yaml \
    --seed 0 --variant natural --passes 1 --limit 32
# -> experiments/target_stats/seed0/natural/pass1/{adapted_model.pt,config.yaml,env.json,adapt_log.jsonl,adapt_metrics.json,checkpoint.sha256}

python -m src.tasks.icassp2027.evaluate \
    --checkpoint experiments/target_stats/seed0/natural/pass1/adapted_model.pt --split dynamic_test --limit 32
```

`--limit` reads from the sample's 64-row `dynamic_adaptation_natural.parquet`/`dynamic_adaptation_balanced.parquet`
and skips the exact-row-count assertion the full-scale run enforces.

## 6. TENT (`adapt_tent.py`)

```bash
python -m src.tasks.icassp2027.adapt_tent --config configs/tent_resnet50.yaml \
    --seed 0 --variant natural --passes 1 --limit 32
python -m src.tasks.icassp2027.evaluate \
    --checkpoint experiments/tent/seed0/natural/pass1/adapted_model.pt --split dynamic_test --limit 32
```

## 7. Entropy + consistency ablation (`adapt_proposed.py`)

Not the paper's PG-STAR method — see the README disclaimer.

```bash
python -m src.tasks.icassp2027.adapt_proposed --config configs/proposed_resnet50.yaml \
    --seed 0 --variant natural --passes 1 --lambda-ent 1.0 --lambda-cons 1.0 --ablation proposed --limit 32
python -m src.tasks.icassp2027.evaluate \
    --checkpoint experiments/proposed/proposed/seed0/natural/pass1/adapted_model.pt --split dynamic_test --limit 32
```

## 8. Caveat: bit-identical reproduction is not guaranteed

Every gradient-based adaptation run (TENT, the entropy+consistency ablation) forwards under bf16
autocast with losses computed in float32, on cuDNN kernels selected at runtime. Re-running any
command above with the same seed and config will **not** in general reproduce bit-identical
checkpoints — only the deterministic, seed-controlled data selection and stream order (splits,
natural/balanced draws, batch order) are guaranteed to reproduce exactly; trained/adapted weights
and metrics derived from them may vary at the level of floating-point non-determinism.

## Disclosure — the internal audit manifest is deliberately not included

The source monorepo keeps a `configs/manifests/internal/audit_all_primary.tsv` reverse map
(`sample_id → source_path, class, adaptation membership, stream position`) for its own audit code.
It is never included in any release, including this one — including it would defeat the
target-label isolation guarantee the adaptation methods rely on (the unlabeled adaptation export
carries no `class_label`/`class_folder_id`/`source_path`/`frame_index` columns; see
`AdaptationParquetDataset` in `src/tasks/icassp2027/data.py`).
