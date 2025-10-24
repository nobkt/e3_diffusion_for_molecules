# Phase 4.1 Usage Guide: Property Validation System

**Document Type**: Usage Guide  
**Phase**: Phase 4.1 (Property Validation System)  
**Date**: 2025-10-24  
**Version**: 1.0

---

## Overview

This guide provides practical instructions for using the **Property Validation System** to train property predictors and validate generated crystals.

The Property Validation System enables:
1. Training a property prediction model from crystal structures
2. Predicting properties of generated crystals
3. Validating that generated crystals match target properties
4. Generating comprehensive validation reports

---

## Quick Start

### 1. Train Property Predictor (30 minutes)

```bash
# Train a property predictor on existing crystal database
python train_property_predictor.py \
    --crystal_db_path data/crystals_with_props.db \
    --property_names bandgap melting_point \
    --exp_name my_predictor \
    --n_epochs 100 \
    --batch_size 32
```

**Output**: 
- `outputs/my_predictor/property_predictor/checkpoint_best.pt`
- Training logs showing loss and MAE per property

### 2. Generate Crystals with Target Properties (5 minutes)

```bash
# Generate crystals with specific target properties
python generate_crystal_with_all_conditions.py \
    --model_path outputs/crystal_model.pt \
    --molecule_db_path data/molecules.db \
    --molecule_id benzene_001 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --n_samples 100 \
    --output_dir generated_samples/
```

**Output**:
- CIF files in `generated_samples/`
- Metadata JSON files

### 3. Validate Generated Crystals (2 minutes)

```bash
# Validate generated crystals against target properties
python validate_generated_crystals.py \
    --predictor_path outputs/my_predictor/property_predictor/checkpoint_best.pt \
    --crystal_dir generated_samples/ \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --output_report validation_report.txt
```

**Output**:
- `validation_report.txt` with detailed statistics
- Console summary of validation results

---

## Detailed Instructions

### Step 1: Data Preparation

The property predictor requires a crystal database with properties (created in Phase 3).

**Check your data**:
```bash
# Verify database exists and has properties
python -c "
from crystal.data.crystal_dataset_with_properties import CrystalDatasetWithProperties
dataset = CrystalDatasetWithProperties(
    db_path='data/crystals_with_props.db',
    property_names=['bandgap', 'melting_point'],
    split='train'
)
print(f'Dataset size: {len(dataset)}')
stats = dataset.get_property_statistics()
print(f'Property statistics: {stats}')
"
```

**Expected output**:
```
Dataset size: 5000
Property statistics: {'mean': [2.5, 180.0], 'std': [1.2, 50.0], ...}
```

### Step 2: Train Property Predictor

#### Basic Training

```bash
python train_property_predictor.py \
    --crystal_db_path data/crystals_with_props.db \
    --property_names bandgap melting_point \
    --exp_name my_predictor \
    --n_epochs 100 \
    --batch_size 32
```

#### Advanced Options

**Custom Model Architecture**:
```bash
python train_property_predictor.py \
    --crystal_db_path data/crystals_with_props.db \
    --property_names bandgap melting_point dielectric_constant \
    --exp_name advanced_predictor \
    --hidden_dim 512 \
    --n_layers 6 \
    --max_neighbors 64 \
    --cutoff_radius 10.0 \
    --n_epochs 200 \
    --batch_size 64 \
    --learning_rate 5e-5
```

**GPU Training**:
```bash
# Automatically uses GPU if available
CUDA_VISIBLE_DEVICES=0 python train_property_predictor.py \
    --crystal_db_path data/crystals_with_props.db \
    --property_names bandgap melting_point \
    --exp_name gpu_predictor \
    --n_epochs 100 \
    --batch_size 128
```

#### Monitoring Training

Training progress is displayed in real-time:

```
Epoch 50/100
  Train Loss: 0.1234
  Val Loss: 0.1456
  Val bandgap MAE: 0.0567
  Val melting_point MAE: 2.3456

Saved best checkpoint to outputs/my_predictor/property_predictor/checkpoint_best.pt
```

**Training Time Estimates** (with GPU):
- 1,000 samples: ~30 minutes
- 10,000 samples: ~5 hours
- 100,000 samples: ~2 days

### Step 3: Validate Generated Crystals

#### Basic Validation

```bash
python validate_generated_crystals.py \
    --predictor_path outputs/my_predictor/property_predictor/checkpoint_best.pt \
    --crystal_dir generated_samples/ \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --output_report validation_report.txt
```

#### Save Detailed Results (JSON)

```bash
python validate_generated_crystals.py \
    --predictor_path outputs/my_predictor/property_predictor/checkpoint_best.pt \
    --crystal_dir generated_samples/ \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --output_report validation_report.txt \
    --output_json validation_results.json
```

#### Custom File Pattern

```bash
# Validate only specific files
python validate_generated_crystals.py \
    --predictor_path outputs/my_predictor/property_predictor/checkpoint_best.pt \
    --crystal_dir generated_samples/ \
    --pattern "crystal_*_final.cif" \
    --target_bandgap 2.5 \
    --output_report validation_report.txt
```

#### Validation Without Targets (Prediction Only)

```bash
# Just predict properties without comparing to targets
python validate_generated_crystals.py \
    --predictor_path outputs/my_predictor/property_predictor/checkpoint_best.pt \
    --crystal_dir generated_samples/ \
    --output_report prediction_report.txt
```

### Step 4: Interpret Validation Reports

#### Understanding the Report

**Report Structure**:
```
================================================================================
CRYSTAL PROPERTY VALIDATION REPORT
================================================================================

Total crystals validated: 100
Properties: bandgap, melting_point

--------------------------------------------------------------------------------
SUMMARY STATISTICS
--------------------------------------------------------------------------------

bandgap:
  Mean Absolute Error (MAE): 0.1234
  Standard Deviation: 0.0567
  Min Error: 0.0123
  Max Error: 0.4567
  Mean Relative Error: 4.93%

melting_point:
  Mean Absolute Error (MAE): 5.6789
  Standard Deviation: 2.3456
  Min Error: 0.5678
  Max Error: 15.6789
  Mean Relative Error: 3.15%
```

**Interpreting Results**:

1. **Mean Absolute Error (MAE)**: Average prediction error
   - **Good**: MAE < 10% of target value
   - **Acceptable**: MAE < 20% of target value
   - **Poor**: MAE > 20% of target value

2. **Mean Relative Error**: Percentage error
   - **Excellent**: < 5%
   - **Good**: 5-10%
   - **Acceptable**: 10-20%
   - **Poor**: > 20%

3. **Standard Deviation**: Consistency of predictions
   - Lower is better (more consistent)

**Example Interpretation**:
- Bandgap MAE of 0.1234 with target 2.5 → **4.9% error → Excellent**
- Melting point MAE of 5.68 with target 180.0 → **3.2% error → Excellent**

#### JSON Output Format

```json
[
  {
    "crystal_path": "generated_samples/crystal_001.cif",
    "predictions": {
      "bandgap": 2.4567,
      "melting_point": 175.6789
    },
    "targets": {
      "bandgap": 2.5,
      "melting_point": 180.0
    },
    "absolute_errors": {
      "bandgap": 0.0433,
      "melting_point": 4.3211
    },
    "relative_errors": {
      "bandgap": 1.73,
      "melting_point": 2.40
    }
  },
  ...
]
```

---

## Python API Usage

### Basic Property Prediction

```python
import torch
from crystal.evaluation.property_predictor import PropertyPredictor
from ase.io import read as ase_read

# Load checkpoint
checkpoint = torch.load('outputs/my_predictor/property_predictor/checkpoint_best.pt')

# Create and load model
predictor = PropertyPredictor(
    property_names=checkpoint['property_names'],
    **checkpoint['model_config']
)
predictor.load_state_dict(checkpoint['model_state_dict'])
predictor.set_normalization_params(
    checkpoint['property_mean'],
    checkpoint['property_std']
)
predictor.eval()

# Load crystal from CIF
atoms = ase_read('crystal.cif')
positions = torch.tensor(atoms.get_scaled_positions()).unsqueeze(0)
cell = torch.tensor(atoms.get_cell().array).unsqueeze(0)
atomic_numbers = torch.tensor(atoms.get_atomic_numbers()).unsqueeze(0)

# Predict properties
with torch.no_grad():
    predictions = predictor(positions, cell, atomic_numbers)

print(f"Predicted bandgap: {predictions['bandgap'].item():.4f}")
print(f"Predicted melting point: {predictions['melting_point'].item():.4f}")
```

### Batch Prediction

```python
import torch
from pathlib import Path

# Load multiple crystals
crystal_files = list(Path('generated_samples').glob('*.cif'))

all_predictions = []
for cif_file in crystal_files:
    atoms = ase_read(cif_file)
    positions = torch.tensor(atoms.get_scaled_positions()).unsqueeze(0)
    cell = torch.tensor(atoms.get_cell().array).unsqueeze(0)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers()).unsqueeze(0)
    
    with torch.no_grad():
        predictions = predictor(positions, cell, atomic_numbers)
    
    all_predictions.append({
        'file': cif_file.name,
        'bandgap': predictions['bandgap'].item(),
        'melting_point': predictions['melting_point'].item(),
    })

# Compute statistics
import numpy as np
bandgaps = [p['bandgap'] for p in all_predictions]
print(f"Mean bandgap: {np.mean(bandgaps):.4f}")
print(f"Std bandgap: {np.std(bandgaps):.4f}")
```

---

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Install dependencies
```bash
pip install torch numpy ase
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size:
   ```bash
   --batch_size 16  # or smaller
   ```

2. Reduce model size:
   ```bash
   --hidden_dim 128 --n_layers 3
   ```

3. Use CPU (slower but no memory limit):
   ```bash
   CUDA_VISIBLE_DEVICES="" python train_property_predictor.py ...
   ```

### Issue: High validation MAE

**Causes**:
1. **Insufficient training data**: Need at least 1,000 samples
2. **Too few epochs**: Try 200-300 epochs
3. **Model too small**: Increase hidden_dim or n_layers
4. **Property not learnable**: Some properties may be too complex

**Solutions**:
```bash
# Longer training with larger model
python train_property_predictor.py \
    --hidden_dim 512 \
    --n_layers 6 \
    --n_epochs 300 \
    ...
```

### Issue: "ValueError: Property X not in CSV"

**Cause**: Property name mismatch between database and CLI

**Solution**: Check available properties:
```python
from crystal.data.crystal_dataset_with_properties import CrystalDatasetWithProperties
dataset = CrystalDatasetWithProperties(
    db_path='data/crystals_with_props.db',
    property_names=None,  # Will show available properties
    split='train'
)
```

### Issue: Predictions are all the same

**Cause**: Model not properly trained or normalized

**Solutions**:
1. Check training loss is decreasing
2. Verify normalization parameters are set
3. Ensure training data has variance in properties

---

## Best Practices

### 1. Training

- **Use at least 1,000 samples** for reliable predictions
- **Train for 100-300 epochs** depending on dataset size
- **Monitor validation MAE** - stop if not improving for 20 epochs
- **Save checkpoints regularly** in case of interruption

### 2. Validation

- **Always use target properties** for meaningful validation
- **Generate at least 100 samples** for statistical significance
- **Check relative errors** not just absolute errors
- **Compare to training data distribution**

### 3. Interpretation

- **MAE < 10%**: Excellent prediction quality
- **MAE 10-20%**: Acceptable for research use
- **MAE > 20%**: May need better predictor or more training data

---

## Performance Guide

### Training Performance

| Dataset Size | Batch Size | GPU | Time per Epoch | Total Time (100 epochs) |
|--------------|------------|-----|----------------|-------------------------|
| 1,000        | 32         | Yes | ~20s           | ~30 min                 |
| 1,000        | 32         | No  | ~2 min         | ~3 hours                |
| 10,000       | 64         | Yes | ~3 min         | ~5 hours                |
| 10,000       | 64         | No  | ~30 min        | ~2 days                 |

### Validation Performance

| Number of Crystals | Validation Time (GPU) | Validation Time (CPU) |
|--------------------|----------------------|----------------------|
| 10                 | ~1s                  | ~5s                  |
| 100                | ~10s                 | ~50s                 |
| 1,000              | ~2 min               | ~10 min              |

---

## Summary

The Property Validation System provides:

✅ **Training**: Train property predictors from crystal databases  
✅ **Prediction**: Predict properties of generated crystals  
✅ **Validation**: Compare predictions to target properties  
✅ **Reporting**: Generate comprehensive validation reports  

**Typical Workflow**:
1. Train predictor (1-5 hours)
2. Generate crystals with target properties (minutes)
3. Validate crystals (seconds-minutes)
4. Analyze validation report

**Success Metrics**:
- Prediction MAE < 10% of target value
- Validation time < 1 second per crystal
- Automated, reproducible validation

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-24  
**For More Information**: See `PHASE4_1_COMPLETION_REPORT_JA.md`
