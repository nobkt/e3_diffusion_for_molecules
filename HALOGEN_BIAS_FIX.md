# Halogen Bias Fix for ASE Databases

## Problem Description

When training molecular generation models on ASE databases with diverse chemical elements (especially those containing halogens like Br, Cl, I), the generated molecules were heavily biased toward halogen atoms instead of reflecting the actual element distribution in the training data.

## Root Cause

The issue was caused by the hardcoded categorical normalization factor of 4 in the diffusion model, which was originally designed for QM9 datasets containing only 5 elements (H, C, N, O, F). When applied to databases with many more elements, the noise added during the diffusion process disproportionately affected rare elements at higher indices, causing them to be sampled much more frequently than they should be.

### Technical Details

1. **Normalization**: Categorical data (atom types) is normalized by dividing by the normalization factor
2. **Noise Addition**: During diffusion, Gaussian noise is added to the normalized values
3. **Denormalization**: Values are multiplied back by the normalization factor
4. **Sampling**: Argmax is applied to determine the final atom type

The problem occurs because:
- With a high normalization factor (4) and many elements (11), normalized values become very small
- When noise is added, it has a disproportionate effect on the rare elements
- After denormalization, rare elements (especially those at higher indices like Br, I) can become dominant

## Solution

The fix implements dynamic normalization factor selection based on the number of unique elements in the dataset:

- **≤ 5 elements**: Use factor 4.0 (original QM9 behavior)
- **6-10 elements**: Use factor 2.0 (moderate reduction)
- **> 10 elements**: Use factor 1.0 (minimal normalization)

## Implementation

### 1. Updated `create_optimal_dataset_config()` in `qm9/general_molecular_db.py`
- Calculates optimal normalization factor based on element count
- Stores recommendation in dataset configuration

### 2. Updated `main_qm9.py`
- Automatically adjusts normalization factors for ASE databases
- Prints information about the adjustment and reasoning

### 3. Updated `suggest_training_parameters()` in `qm9/general_molecular_db.py`
- Includes normalization factor recommendations in training suggestions
- Provides explanatory text about the fix

### 4. Updated `update_ase_dataset_config()` in `qm9/dataset.py`
- Computes actual element statistics during dataset loading
- Includes normalization recommendations

## Usage

The fix is automatically applied when using ASE databases. Users will see output like:

```
Adjusted categorical normalization factor for ASE database:
  Original factors: [1, 4, 1]
  Updated factors: [1, 1.0, 1]
  Reason: Dataset has 11 elements
```

## Training Command Example

For a database with 11 elements (like the one in the original problem):

```bash
python main_qm9.py \
  --dataset ase_db \
  --ase_db_path your_database.db \
  --batch_size 16 \
  --n_epochs 200 \
  --nf 256 \
  --n_layers 8 \
  --lr 5e-05 \
  --normalize_factors [1, 1.0, 1] \
  --conditioning molecular_weight pi_conjugation_ratio
```

## Verification

The fix has been tested with synthetic databases that reproduce the halogen bias scenario. The system now correctly identifies high element diversity and applies appropriate normalization factors to prevent bias.

## Benefits

1. **Eliminates halogen bias**: Generated molecules now respect the actual element distribution from training data
2. **Automatic adjustment**: No manual parameter tuning required
3. **Backward compatible**: QM9 and other small-element datasets continue to use original settings
4. **Informative output**: Clear explanations of why adjustments are made

This fix ensures that molecular generation models trained on diverse chemical databases produce realistic and unbiased molecular structures.