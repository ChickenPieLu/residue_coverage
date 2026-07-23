# Residue Coverage in the Cornfield

## Current Model

The current model is an SMP U-Net with an ImageNet-pretrained ResNet-34
encoder. Its training code is in the repository root and uses A/B/C for
training, D for validation and E only for the final test.

```bash
python main.py
python test.py
python visualiseE.py
```

The best validation checkpoint is saved as
`smp_unet_resnet34_imagenet_abc_bce_dice_seed42.pth`.

## Project Layout and Commands

The historical Mini U-Net and Random Forest approaches live under `legacy/`
and use the same
location-based split:

- training: `A`, `B`, `C`
- validation: `D`
- final test: `E`

The Mini U-Net code is in `legacy/unet/`:

```bash
python -m legacy.unet.main
python -m legacy.unet.test
python -m legacy.unet.visualiseE
```

The Random Forest code is in `legacy/classical_ml/`. Its default training
command trains on A/B/C and validates on D; the test command evaluates only
the unseen E images:

```bash
python -m legacy.classical_ml.training
python -m legacy.classical_ml.test
python -m legacy.classical_ml.visualiseE
```

Both visualisation scripts use the same three E cases, so their masks and
coverage errors can be compared directly. The Random Forest keeps its
historical probability threshold of `0.60`; it can be overridden with
`--threshold`.

## U-Net Experiments

This document records the development and evaluation of a U-Net model for
crop residue segmentation and coverage estimation.

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
several locations and validated on a different location. Location D is used
for early stopping and model selection, so it is treated as a validation set
rather than a completely unbiased final test set.

### Shared Experimental Configuration

| Setting | Value |
|---|---|
| Model | Mini U-Net |
| Input | RGB image |
| Output | Single-channel segmentation logits |
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

The best checkpoint in each experiment was selected using IoU on validation
location D. Location E was not used for training, model selection or
hyperparameter tuning.

## Experiment 1: BCE Baseline

### Loss Function

The baseline uses binary cross-entropy with logits:

`BCEWithLogitsLoss`

### Results

The best checkpoint was obtained at epoch 24. Training stopped at epoch 34
after 10 consecutive epochs without improvement.

| Dataset | Loss | IoU | Dice | Precision | Recall | Coverage MAE | True coverage | Predicted coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training (A/B/C) | 0.2307 | 0.6341 | 0.7761 | 0.7820 | 0.7702 | 0.0410 | 22.61% | 22.27% |
| Validation (D) | 0.3522 | 0.5168 | 0.6814 | 0.6420 | 0.7260 | 0.0527 | 22.25% | 25.16% |

### Interpretation

The baseline achieves an IoU of `0.5168` and a Dice score of `0.6814` on
validation location D. This indicates that the model has learned features
that generalise across locations, although a performance gap remains between
the training and validation sets.

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
improvement through better loss functions, data augmentation and additional
cross-location data.

### Reproducibility

A repeated run using the same random seed and an explicitly seeded training
DataLoader produced nearly the same best validation IoU as the previous run
(`0.5168` versus `0.5183`). The reproducible result of `0.5168` is used as
the official BCE baseline.

## Experiment 2: BCE + Dice Loss

### Motivation and Loss Function

The second experiment tests whether directly optimising the overlap between
the predicted and true masks improves segmentation performance. The loss is
an unweighted sum of BCE and soft Dice loss:

`total loss = BCE loss + Dice loss`

BCE supervises each pixel independently, while Dice loss gives greater
emphasis to the overlap of the complete predicted residue region with the
ground-truth region.

All other settings, including the data split, random seed, optimiser,
learning rate, batch size, threshold and early-stopping procedure, were kept
the same as in the BCE baseline.

### Training Behaviour

At epoch 0, the model predicted only background on location D, producing an
IoU of `0.0000`. Validation IoU then increased rapidly to `0.4195` by epoch 4
and continued to improve more slowly. New best checkpoints were obtained at
epochs 9, 10, 13, 16, 20 and 25.

The best validation IoU was obtained at epoch 25. Training stopped at epoch
35 after 10 consecutive epochs without improvement. After epoch 25, the
training loss continued to fluctuate around a gradually declining level, but
validation IoU remained approximately between `0.46` and `0.52`, indicating
that additional training was not producing a consistent improvement in
cross-location generalisation.

### Results

| Dataset | Total loss | IoU | Dice | Precision | Recall | Coverage MAE | True coverage | Predicted coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training (A/B/C) | 0.5276 | 0.6517 | 0.7892 | 0.7991 | 0.7795 | 0.0367 | 22.61% | 22.06% |
| Validation (D) | 0.8274 | 0.5163 | 0.6810 | 0.6332 | 0.7367 | 0.0521 | 22.25% | 25.89% |

At the best epoch, the final training-batch averages reported during training
were approximately:

| Component | Value |
|---|---:|
| BCE loss | 0.2480 |
| Dice loss | 0.3065 |
| Total loss | 0.5546 |

These batch-averaged values are included only to show the relative
contribution of the two loss components during training. The checkpoint
metrics above were calculated by evaluating the saved best model over the
complete datasets.

### Interpretation

The BCE + Dice model achieves a validation IoU of `0.5163`, which is almost
identical to the BCE baseline IoU of `0.5168`. Dice score is also essentially
unchanged (`0.6810` versus `0.6814`). Therefore, this experiment provides no
evidence that adding Dice loss materially improves cross-location
segmentation under the current configuration.

Compared with BCE alone, BCE + Dice produces slightly lower precision and
slightly higher recall. It predicts more residue on location D, increasing
aggregate predicted coverage from `25.16%` to `25.89%`. The per-image coverage
MAE improves only marginally, from `5.27` to `5.21` percentage points.

The BCE + Dice training and validation IoU difference is:

`0.6517 - 0.5163 = 0.1354`

This is larger than the BCE baseline gap of `0.1173`. The combination of a
higher training IoU and unchanged validation IoU suggests that the main
remaining limitation is generalisation to a new location rather than an
inability to fit the training data.

The absolute loss values from the two experiments must not be compared
directly. The BCE + Dice loss is the sum of two terms, so a total loss around
`0.5` does not imply that the model is undertrained, nor does it mean that it
has a fixed amount of improvement remaining. Overfitting is assessed using
the training-validation performance gap and the validation trend, not by how
close the training loss is to zero.

## Comparison of Loss Functions

| Validation metric | BCE | BCE + Dice | Change |
|---|---:|---:|---:|
| IoU | 0.5168 | 0.5163 | -0.0005 |
| Dice | 0.6814 | 0.6810 | -0.0004 |
| Precision | 0.6420 | 0.6332 | -0.0088 |
| Recall | 0.7260 | 0.7367 | +0.0107 |
| Coverage MAE | 0.0527 | 0.0521 | -0.0006 |
| Predicted coverage | 25.16% | 25.89% | +0.73 pp |

The observed differences are extremely small and could be caused by normal
training variation. A multi-seed comparison would be required to make a
strong claim about which loss is better. For the present project, the main
conclusion is that replacing BCE with an unweighted BCE + Dice objective did
not overcome the cross-location generalisation bottleneck.

## Current Conclusion and Wrap-up Plan

The project has now established a reproducible U-Net training and evaluation
pipeline with:

1. location-based training, validation and test splits;
2. deterministic random seeds and DataLoader shuffling;
3. IoU-based checkpoint selection and early stopping;
4. IoU, Dice, precision, recall and coverage-error evaluation;
5. a reproducible BCE baseline;
6. a controlled BCE + Dice loss comparison.

Simply increasing the maximum epoch count is unlikely to provide a meaningful
improvement. The BCE + Dice run already shows a training-validation gap, and
validation IoU stopped improving consistently after epoch 25 even though the
training objective continued to decrease. Increasing patience slightly could
allow for more validation fluctuation, but it would not address the main
cross-location limitation.

The most valuable source of further improvement would be additional real data
from a wider range of locations, soil colours, lighting conditions, residue
types and camera conditions. Since such data is not currently available, the
remaining training work will be limited to one controlled data-augmentation
experiment.

### Final Planned Experiment

The final experiment will use modest transformations that preserve the
segmentation labels:

- random horizontal flipping;
- random vertical flipping;
- random rotation by 0, 90, 180 or 270 degrees;
- optionally, small brightness and contrast adjustments applied only to the
  RGB image.

The image and mask must receive exactly the same geometric transformation.
Colour or brightness transformations must be applied only to the image.

Data augmentation cannot create genuinely new locations or soil and lighting
conditions, so only a small improvement is expected. A plausible improvement
range is approximately `0.00–0.04` IoU, and no improvement is also possible.
The purpose of this final experiment is to test whether simple invariances
reduce overfitting, rather than to begin an extensive new tuning cycle.

After this experiment, development will move to project wrap-up:

1. compare the Random Forest, BCE U-Net, BCE + Dice U-Net and augmented U-Net;
2. visualise representative successful and failed predictions;
3. analyse cross-location failure cases and coverage-estimation errors;
4. document limitations and possible future work;
5. evaluate the selected final configuration once on untouched location E.

Location E must remain untouched until the final model configuration has been
selected. Its result will be reported as the final held-out test result and
must not be used for further hyperparameter tuning.
