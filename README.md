# residue coverage in the cornfield

## First stage

### Baseline

- Model: MiniUNet
- Loss: BCEWithLogitsLoss
- Training locations: *A, B, C*
- Validation location: *D*
- Epochs: 20
- Threshold: 0.5
- Training IoU: **0.6164**
- Validation IoU: **0.4721**
- Validation Dice: approximately 0.641
- Validation coverage MAE: approximately 5.83 percentage points
