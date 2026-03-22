# VigilXAI

A multi-phase chest X-ray classification system exploring the full arc of medical AI development — from binary pneumonia detection to explainable multi-label pathology classification.

**Author:** Aaron Gauntt  
**Status:** Phase 2 complete — Phase 3 in progress

---

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Binary pneumonia classifier | Complete |
| 2 | Multi-label pathology detection | Complete |
| 4 | Grad-CAM explainability | In Progress |

---

## Phase 1 — Binary pneumonia classification

### Architecture

ResNet34 pretrained on ImageNet, fine-tuned for binary classification. Dropout (p=0.5) before the output layer. AdamW optimizer with cosine annealing scheduler and early stopping on validation AUROC.

### Dataset

Kaggle chest X-ray dataset — ~5,800 training images, 3:1 pneumonia-to-normal class imbalance. Class imbalance addressed through weighted random sampling and data augmentation.

**Augmentation:** horizontal flip, rotation (±10°), affine translation, color jitter, sharpness adjustment — all clinically justified for chest X-ray variability.

### Results

| | Baseline | Run 1 |
|---|---|---|
| Test AUC | 0.9515 | 0.9601 |
| Normal recall | 0.4615 | 0.7308 |
| Missed pneumonia | 0 | 4 |
| False positives | 126 | 63 |
| Accuracy | 80.3% | 89.3% |

Normal recall improved 27 percentage points after weighted sampling and augmentation. The tradeoff of 4 missed pneumonia cases against 63 fewer false alarms reflects a deliberate shift toward a more balanced clinical operating point.

---

## Phase 2 — Multi-label pathology detection

### Architecture

DenseNet121 pretrained on ImageNet — the architecture used in Stanford's original CheXNet paper. Classifier head replaced with Dropout (p=0.5) + Linear(1024 → 15). BCEWithLogitsLoss with sigmoid activations at inference. Per-label AUROC reported via `MultilabelAUROC`.

### Dataset

NIH ChestX-ray14 — 112,120 frontal chest X-rays from 30,805 patients across 15 pathology labels. PA views only, filtered to reduce noise from mixed AP/PA imaging conditions.

**Patient-level splitting** — all train/val/test splits (70/15/15) performed at the patient ID level to prevent the model from learning patient-specific anatomical shortcuts rather than pathological features.

**Multi-label weighted sampling** — samples weighted by the sum of inverse-sqrt label frequencies across positive conditions. Sqrt smoothing prevents extreme oversampling of rare conditions that caused overfitting in earlier runs.

### Results

Best test mean AUROC: **0.8056**

| Label | AUC | Positives |
|---|---|---|
| Hernia | 0.9423 | 23 |
| Cardiomegaly | 0.9236 | 276 |
| Effusion | 0.8802 | 953 |
| Emphysema | 0.8785 | 239 |
| Pneumothorax | 0.8693 | 542 |
| Edema | 0.8619 | 41 |
| Mass | 0.8469 | 552 |
| Atelectasis | 0.8104 | 829 |
| Pleural Thickening | 0.7746 | 352 |
| Consolidation | 0.7666 | 233 |
| No Finding | 0.7578 | 5811 |
| Nodule | 0.7487 | 684 |
| Fibrosis | 0.7392 | 225 |
| Pneumonia | 0.6612 | 84 |
| Infiltration | 0.6229 | 1340 |
| **Mean** | **0.8056** | |

Weak Infiltration (0.6229) and Pneumonia (0.6612) AUC are consistent with published NIH baselines and reflect label noise inherent to NLP-extracted annotations rather than model failure. Infiltration was subsequently removed from CheXpert's label schema for this reason. Phase 3 migrates to CheXpert Plus for higher-quality labels.

---

## Repository structure

```
vigilxai/
├── src/
│   ├── architectures.py    # ResNet18, ResNet34, DenseNet121
│   ├── data.py             # NIHChestDataset, transforms, patient-level splits
│   ├── train.py            # training loop with early stopping
│   ├── eval.py             # per-label AUROC evaluation
│   ├── debug.py            # data validation utilities
│   └── utils.py            # seed setting, directory helpers
├── configs/
│   ├── resnet18.yaml
│   ├── resnet34.yaml
│   └── densenet121.yaml
└── outputs/                # checkpoints — gitignored
```

---

*VigilXAI is a portfolio project. Not intended for clinical use.*
