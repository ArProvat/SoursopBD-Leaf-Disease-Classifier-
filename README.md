# SoursopBD Leaf Disease Classifier


A convolutional neural network trained **from scratch** to classify soursop leaf diseases into six categories, running end-to-end on a free-tier Google Colab T4 GPU.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [SoursopBD — Mendeley Data v2](https://data.mendeley.com/datasets/hjrhrt5hs8/2) |
| Total images | 3,838 (after deduplication — see note below) |
| Image size | 1,024 × 1,024 JPG, resized to 224 × 224 for training |
| Classes | 6 |
| Class balance | Near-perfect (1.1× imbalance ratio) |

**Classes:** Cutting Caterpillar · Cutting Weevil · Die Back · Healthy · White Fly · Yellow

### Data integrity note

The raw dataset zip extracts into a nested folder structure that causes a naive `os.walk` to load every image twice (7,676 paths from 3,838 unique files). The notebook applies MD5 hash-based deduplication before any splitting, reducing the manifest to 3,838 unique images. Without this step the test set leaks into training, producing an artificially inflated 100% accuracy. The trustworthy result — reported here — comes from the deduplicated pipeline.

---

## Approach

### 1. Data pipeline

Images are loaded and normalised to `[0, 1]` float in `tf.data`. No augmentation runs inside `tf.data` — this avoids `tf.function` graph-tracing crashes caused by dynamic shapes and Keras stateful layers in parallel `map()` calls.

Augmentation lives as Keras preprocessing layers at the **top of the model**, where it:
- Runs on GPU as part of the compute graph
- Activates automatically during `model.fit()` (`training=True`)
- Is a no-op during `model.predict()` and `model.evaluate()` (`training=False`)
- Requires no extra toggle code

### 2. Augmentation choices

| Transform | Reason | Constraint applied |
|---|---|---|
| Random horizontal/vertical flip | Leaves have no canonical orientation | — |
| Random rotation ±30° | Branch angle varies in field photos | `fill_mode="nearest"` — no black borders |
| Random zoom ±10% | Field shooting distance varies | — |
| Random brightness ±0.2 | Field lighting varies significantly | Capped at 0.2 — 0.3 pushed dark pixels to black |
| Random contrast ±0.3 | Shadow and overexposure variation | — |
| Hue / saturation | **Not applied aggressively** | Yellow and brown colouration **is** the disease signal |

Normalisation (ImageNet statistics) is placed **after** augmentation inside the model. Colour augmentation ops clip to `[0, 1]` — applying them after normalisation (where values are `~[-2, +2]`) destroys the image, producing solid grey outputs.

### 3. Architecture

A custom lightweight ResNet-style CNN — no pretrained weights, no imported model from `tf.keras.applications`.

```
Input  224 × 224 × 3  (float32, [0, 1])
  Augmentation head  (training only)
  NormaliseLayer     (x - mean) / std  →  ~[-2, +2]
  Stem    Conv7×7(64) → BN → ReLU → MaxPool     64 × 56 × 56
  Layer1  2 × ResBlock(64,  stride=1)            64 × 56 × 56
  Layer2  2 × ResBlock(128, stride=2)           128 × 28 × 28
  Layer3  2 × ResBlock(256, stride=2)           256 × 14 × 14
  Layer4  2 × ResBlock(512, stride=2)           512 ×  7 ×  7
  GlobalAveragePooling2D                         512
  Dropout(0.4)
  Dense(6)  →  raw logits
```

**Total parameters: ~11.2 million**

Key design decisions:

- **Residual (skip) connections** — allow gradients to bypass conv layers via an identity path, preventing vanishing gradients when training deep networks from random initialisation.
- **Global Average Pooling** — collapses `7×7×512 → 512` with zero parameters, compared to `Flatten + Dense` which would add ~12.8M parameters — guaranteed overfitting on 3,000 images.
- **BatchNorm momentum = 0.9** — TF uses the decay convention (opposite to PyTorch). `TF 0.9 = PyTorch 0.1`, meaning 10% update per batch. Using `0.99` (a common mistake) updates only 1% per batch — running statistics never converge and validation loss explodes.
- **`use_bias=False` on Conv layers** — BatchNorm's learnable `beta` parameter absorbs any Conv bias, making it redundant.

### 4. Training

| Component | Choice | Reason |
|---|---|---|
| Loss | `SparseCategoricalCrossentropy(from_logits=True)` | Integer labels; numerically stable |
| Optimiser | Adam, lr=1e-3, clipnorm=1.0 | Fast convergence without LR tuning; gradient clipping prevents early instability |
| LR scheduler | `ReduceLROnPlateau(factor=0.5, patience=7)` | Automatic; no hand-designed schedule needed |
| Early stopping | patience=15, `restore_best_weights=True` | Prevents overfitting; auto-restores best checkpoint |
| Class weights | Computed, passed to `model.fit()` | ~1.0 per class on this balanced dataset — no effect, but correct practice |

**Split:** stratified 80 / 10 / 10 (train / val / test), test held out before val split.

Training ran for approximately 75 epochs before early stopping triggered. The LR scheduler fired twice (epochs ~18 and ~44), with each cut followed by measurable improvement in validation loss.

---

## Results

**Primary metric: Macro F1** — chosen before training, not post-hoc. With class imbalance, accuracy rewards predicting the majority class. Macro F1 weights all six classes equally regardless of frequency.

### Test set (384 images, deduplicated, never seen during training)

| Metric | Score |
|---|---|
| **Macro F1** | **0.9974** |
| Macro Precision | 0.9974 |
| Macro Recall | 0.9974 |
| Accuracy | 99.7% |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cutting Caterpillar | 0.98 | 1.00 | 0.99 | 63 |
| Cutting Weevil | 1.00 | 0.98 | 0.99 | 64 |
| Die Back | 1.00 | 1.00 | 1.00 | 65 |
| Healthy | 1.00 | 1.00 | 1.00 | 66 |
| White Fly | 1.00 | 1.00 | 1.00 | 64 |
| Yellow | 1.00 | 1.00 | 1.00 | 62 |

**Single error:** one Cutting Weevil leaf predicted as Cutting Caterpillar. Both classes involve physical cutting damage by insects — the most semantically reasonable confusion the model can make.

### Why the results are high — and what limits confidence

The dataset is exceptionally clean for a field collection: near-perfect class balance, white-background photos with controlled lighting, and clear large-scale visual differences between classes (green intact leaves vs. yellow discolouration vs. white powdery deposits vs. physical cutting damage). This is not a hard computer vision problem — it is a tractable classification problem with well-separated classes.

The 95% confidence interval on 99.7% accuracy over 384 test images is approximately **[0.988, 0.999]** — a wide band. A larger test set would tighten this.

---

## Limitations and what I would do next

**Limitations**

- Dataset collected in two cities in Bangladesh under controlled conditions. Performance on field photos from different regions, camera types, or lighting conditions is unknown (domain shift).
- No out-of-distribution detection — the model will confidently classify anything as one of the six classes, including non-soursop leaves or unseen diseases.
- Test set size (384 images, ~64 per class) is thin for strong statistical conclusions.

**With more time**

| Improvement | Expected impact |
|---|---|
| Test-Time Augmentation (TTA) | Average predictions over N augmented views → typically +1–2% F1 |
| Label smoothing (ε=0.1) | Prevents overconfident predictions; better calibration |
| LR warmup over first 5 epochs | Stabilises early training from random initialisation |
| Reduce channel width (32→64→128→256) | ~3M parameters — may generalise better on 3K images |
| Independent held-out test set | Collected at different time/location for honest domain evaluation |

---

## How to run

1. Open `soursop_classifier.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all: **Runtime → Run all**

The notebook downloads the dataset automatically via the Mendeley public API, trains, evaluates, and saves all plots to `/content/`. Expected runtime: 90–120 minutes on a T4.

---

## AI tools disclosure

**Claude (Anthropic)** was used to reason through architectural decisions (residual blocks vs. plain CNN, GAP vs. Flatten, Adam vs. SGD), debug framework-specific issues (TF BatchNorm momentum convention, augmentation placement relative to normalisation, `tf.data` graph-tracing crashes), and structure the overall approach. All code, implementation decisions, and written analysis in this repository were produced by the author.

---

## Repository structure

```
├── soursop_classifier.ipynb   # Main Colab notebook — run this
└── README.md                  # This file
```
