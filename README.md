# residue coverage in the cornfield

## U-Net Baseline

This experiment establishes the first reproducible U-Net baseline for crop
residue segmentation and coverage estimation.

### Task

Given an RGB field image, the model predicts a binary segmentation mask:

- `1`: crop residue
- `0`: background

The predicted residue coverage is calculated as the proportion of pixels
classified as crop residue.

### Data Split

The dataset was split by location rather than by randomly mixing images:

- Training set: locations A, B and C
- Validation set: location D
- Final test set: location E (kept completely unseen)

This is a cross-location evaluation: the model is trained on images from
several locations and validated on a different location.

### Baseline Configuration

| Setting | Value |
|---|---|
| Model | Mini U-Net |
| Input | RGB image |
| Output | Single-channel segmentation logits |
| Loss function | Binary Cross-Entropy with Logits |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 4 |
| Random seed | 114514 |
| Training DataLoader | Shuffled with a separately seeded generator |
| Validation DataLoader | Not shuffled |
| Prediction threshold | 0.5 |
| Maximum epochs | 50 |
| Early-stopping patience | 10 epochs |
| Model-selection metric | Validation IoU |

The best checkpoint was selected using IoU on validation location D.
Location E was not used for training, model selection or hyperparameter
tuning.

### Baseline Results

The best checkpoint was obtained at epoch 24. Training stopped at epoch 34
after 10 consecutive epochs without improvement.

| Dataset | Loss | IoU | Dice | Precision | Recall | Coverage MAE | True coverage | Predicted coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training (A/B/C) | 0.2307 | 0.6341 | 0.7761 | 0.7820 | 0.7702 | 0.0410 | 22.61% | 22.27% |
| Validation (D) | 0.3522 | 0.5168 | 0.6814 | 0.6420 | 0.7260 | 0.0527 | 22.25% | 25.16% |

### Interpretation

The baseline achieves an IoU of `0.5168` and a Dice score of `0.6814` on
the unseen validation location D. This indicates that the model has learned
features that generalise across locations, although a performance gap remains
between the training and validation sets.

Validation recall (`0.7260`) is higher than precision (`0.6420`). Therefore,
the model detects most labelled residue pixels but also classifies some
background pixels as residue.

The predicted aggregate coverage on location D is `25.16%`, compared with a
true aggregate coverage of `22.25%`, corresponding to an overall
overestimation of `2.91` percentage points. However, the per-image coverage
MAE is `5.27` percentage points, showing that individual-image errors are
larger than the aggregate difference because overestimation and
underestimation partially cancel across images.

The training and validation IoU difference is:

`0.6341 - 0.5168 = 0.1173`

This suggests moderate cross-location generalisation error, leaving room for
improvement through better loss functions, data augmentation and model
development.

### Reproducibility

A repeated run using the same random seed and an explicitly seeded training
DataLoader produced nearly the same best validation IoU as the previous run
(`0.5168` versus `0.5183`). The reproducible result of `0.5168` is used as
the official BCE baseline.

### Planned Experiments

Future experiments will change one component at a time while keeping the
same data split and evaluation procedure:

1. BCE + Dice loss
2. Data augmentation
3. Learning-rate adjustment or scheduling
4. Model architecture improvements
5. Prediction-threshold analysis

All experiments will continue to use location D for validation. Location E
will remain untouched until the final model configuration has been selected.