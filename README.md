# SPAD-TTA: Benchmark & Data Inspection

> **Official Supplementary Repository for ICASSP 2027 Submission**  
> **"PG-STAR: Purity-Gated Test-Time Adaptation for Dynamic SPAD Classification in Machine Vision"**  
> *Jui-Huang Tsai, Yi-Ching Hung, Jia-Yu Lin, and Chen-Yi Lee*  
> *Institute of Electronics, National Yang Ming Chiao Tung University (NYCU), Taiwan*

![Static vs. dynamic SPAD acquisition and dual-stream benchmark](docs/figures/fig1_wide.png)
![PG-STAR architecture overview](docs/figures/fig2_pg_star_arch.png)

This repository provides the **SPAD-Electronics sample dataset, dual-stream benchmark protocol, and runnable baseline pipelines** for reviewers to inspect the dataset properties and evaluate the test-time adaptation (TTA) setup.

---

## 📌 Benchmark Motivation & Key Insights

High-speed robotic vision with Single-Photon Avalanche Diode (SPAD) sensors enables microsecond-level exposure without conventional motion blur. However:
1. **Visual Sharpness ≠ Domain Invariance**:
   Under passive ambient light (800 lux), SPAD frames of components spinning at 100 Hz remain visually sharp (Fig. 1 top). Yet, a standard ResNet-50 classifier trained on stationary (0 Hz) frames suffers a **30.7% error rate** on dynamic frames due to acquisition and near-sensor distribution shifts.
2. **The Stream Order Dilemma in Online TTA**:
   Deploying conventional TTA methods reveals an acute arrival-order sensitivity. Methods like TENT collapse to ~86% error on temporal class-correlated streams, while graph-based LAME deteriorates to 71.2% on i.i.d. streams. To assess deployment safety, **SPAD-TTA evaluates methods by worst-stream error ($E_{\text{worst}}$)** across both stream orderings.

---

## 🔍 Dataset & Data Inspection

The full **SPAD-Electronics** dataset comprises **45,851 frames** (15,424 static + 30,427 dynamic) across 20 electronic component classes, recorded using an in-house $32 \times 32$ SPAD sensor array with TDC readout under constant 800 lux illumination (each frame integrates 40 photon-count exposures, stored as $640 \times 640$ grayscale PNGs):

| ID | Class Name | ID | Class Name | ID | Class Name | ID | Class Name |
|:--:|:-----------|:--:|:-----------|:--:|:-----------|:--:|:-----------|
| 1 | Inductor | 6 | Oscillator | 11 | 3-pin RA Header | 16 | Optical Switch A |
| 2 | Resistor | 7 | 4-pin RA Header | 12 | Straight Header | 17 | Optical Switch B |
| 3 | LED | 8 | 5-pin Housing | 13 | Vibration Switch | 18 | Photointerrupter |
| 4 | Film Capacitor | 9 | USB-C Receptacle | 14 | Shock Sensor | 19 | Photodiode |
| 5 | Tactile Switch | 10 | Humidity Sensor | 15 | Electrolytic Cap. | 20 | Passive Buzzer |

### Bundled Reviewer Sample Structure
To facilitate lightweight inspection and pipeline verification, this repository includes a curated sample across all 20 classes (~290 static frames + ~260 dynamic frames + 128 unlabeled streaming instances):

```text
data/
└── SPAD_Electronics/
    ├── 0_HZ/                       # Static condition (0 Hz, part at rest)
    │   └── <class_id>/             # Classes 1–20
    │       ├── direction_1/ ...    # Training viewpoints (Directions 1–3: 9,733 frames in full dataset)
    │       ├── direction_4/        # Validation viewpoint (2,765 frames in full dataset)
    │       └── direction_5/        # Test viewpoint (2,926 frames in full dataset)
    └── 100_HZ/                     # Dynamic condition (100 Hz motorized spinning stage)
        └── <class_id>/             # Sequential capture frames (photoNNNN.png)

data_exports/
└── spad_electronics_hf/            # Sample of the unlabeled adaptation stream (Parquet format)
    ├── dynamic_adaptation_natural.parquet
    └── dynamic_adaptation_balanced.parquet
```

### Inspecting the Data in Python
Reviewers can quickly verify the image format, metadata, and parquet streaming layout:
```python
import pyarrow.parquet as pq
from PIL import Image

# 1. Inspect static (rest) vs dynamic (100 Hz spinning) images
static_img = Image.open("data/SPAD_Electronics/0_HZ/1/direction_1/photo0.png")
dynamic_img = Image.open("data/SPAD_Electronics/100_HZ/1/photo1118.png")
print(f"Static size: {static_img.size}, Dynamic size: {dynamic_img.size}")

# 2. Inspect the unlabeled dynamic adaptation stream
table = pq.read_table("data_exports/spad_electronics_hf/dynamic_adaptation_natural.parquet")
print(f"Stream preview ({table.num_rows} samples), columns: {table.schema.names}")
```

---

## ⚙️ Dual-Stream Benchmark Protocol

Both streams replay **an identical multiset of 4,800 dynamic frames** (150 batches, batch size $B=32$); the arrival permutation $\pi$ is the sole variable:
* **Temporal Stream ($S_{\text{temp}}$)**: Preserves capture-index order ($\sim 240$ frames per class block, median batch purity 1.00). Mimics conveyor/robotic inspection lines.
* **i.i.d. Stream ($S_{\text{iid}}$)**: A fixed global permutation of the exact same frames (interleaved classes, median batch purity 0.12). Mimics the standard i.i.d. assumption used in typical TTA benchmarks.

The evaluation metric is **worst-stream error**:
$$E_{\text{worst}} = \max\left(E(S_{\text{temp}}), E(S_{\text{iid}})\right)$$

---

## 🚀 Quickstart: 1-Minute Smoke Test

The commands below use `--smoke` / `--limit` to run end-to-end against the bundled sample data (run them from the repo root, in order):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Source-only ResNet-50 pipeline check (2 epochs).
#    --out puts the checkpoint where step 3 expects it (--smoke alone writes to experiments/source_only/_smoke/).
python -m src.tasks.icassp2027.train_source --config configs/source_only_resnet50.yaml --seed 0 --smoke \
    --out experiments/source_only/seed0

# 3. TENT test-time adaptation pipeline check on the unlabeled sample stream (starts from the step-2 checkpoint)
python -m src.tasks.icassp2027.adapt_tent --config configs/tent_resnet50.yaml --seed 0 --variant natural --limit 32
```

For full reproduction commands and diagnostic baselines (e.g., LODO viewpoint generalization, BN statistics adaptation), see [docs/reproducibility_commands.md](docs/reproducibility_commands.md).

---

## 📦 Release Roadmap

* **Reviewer Release (Current)**:
  * ✅ Representative static & dynamic SPAD-Electronics sample data across all 20 classes.
  * ✅ Full dataset splitting protocol (`configs/splits.yaml`).
  * ✅ Core adaptation baselines (Source-only, BN Target-Stats, TENT, LODO diagnostic).
  * ✅ Dual-stream evaluation harness and streaming data loaders.
* **Post-Acceptance Release (Camera-Ready)**:
  * 🔜 Full 45,851-frame SPAD-Electronics dataset download package.
  * 🔜 Pre-trained ResNet-50 model checkpoints across all seeds.
  * 🔜 Full benchmark evaluation scripts and PG-STAR module package.

---

## 📝 Citation

```bibtex
@inproceedings{tsai2027pgstar,
  title     = {PG-STAR: Purity-Gated Test-Time Adaptation for Dynamic SPAD Classification in Machine Vision},
  author    = {Tsai, Jui-Huang and Hung, Yi-Ching and Lin, Jia-Yu and Lee, Chen-Yi},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## 📄 License

Code and sample data in this repository are released under **CC BY-NC 4.0** (Academic / Non-Commercial Use).
