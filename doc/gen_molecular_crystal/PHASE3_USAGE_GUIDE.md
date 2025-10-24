# Property-Conditioned Crystal Generation - Usage Examples

This document provides practical examples for using the Phase 3 implementation of property-conditioned molecular crystal generation.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Training](#training)
3. [Generation](#generation)
4. [Complete Workflow Example](#complete-workflow-example)

---

## Data Preparation

### Step 1: Prepare Property Data CSV

Create a CSV file with crystal properties. The CSV must have a `crystal_id` column matching IDs in your crystal database.

Example `crystal_properties.csv`:
```csv
crystal_id,bandgap,melting_point,dielectric_constant
crystal_001,2.5,180.0,3.2
crystal_002,2.8,195.0,3.5
crystal_003,2.2,165.0,2.8
...
```

### Step 2: Merge Properties with Crystal Database

Use the data preparation script to merge properties into your crystal database:

```bash
python scripts/prepare_property_dataset.py \
    --crystal_db data/crystals.db \
    --property_csv data/crystal_properties.csv \
    --output_db data/crystals_with_properties.db \
    --property_names bandgap melting_point dielectric_constant \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --validate
```

**Arguments:**
- `--crystal_db`: Path to input crystal ASE database
- `--property_csv`: Path to CSV with property data
- `--output_db`: Path for output database with properties
- `--property_names`: List of property names to include
- `--train_ratio/val_ratio/test_ratio`: Dataset split ratios
- `--validate`: Run validation checks before merging

**Output:**
- `crystals_with_properties.db`: Database with merged properties
- `property_statistics.json`: Property statistics (mean, std, min, max)
- `dataset_splits.json`: Train/validation/test split indices

### Step 3: Verify Data

The script automatically validates:
- ✅ All crystals have corresponding properties
- ✅ No NaN or missing values
- ✅ No zero-variance properties
- ✅ All required columns exist

---

## Training

### Basic Training Command

Train a model with property conditioning:

```bash
python main_crystal_with_properties.py \
    --crystal_db_path data/crystals_with_properties.db \
    --molecule_db_path data/molecules.db \
    --property_names bandgap melting_point dielectric_constant \
    --exp_name crystal_bandgap_meltingpoint_dielectric \
    --n_epochs 200 \
    --batch_size 32 \
    --lr 1e-4 \
    --conditioning_dim 256 \
    --property_hidden_dim 512 \
    --property_n_layers 3 \
    --condition_on_property True \
    --condition_on_molecule True \
    --condition_on_space_group False \
    --condition_on_density False
```

**Key Arguments:**

**Data:**
- `--crystal_db_path`: Path to crystal database with properties
- `--molecule_db_path`: Path to molecule database (for molecular features)
- `--property_names`: List of properties to condition on

**Model Architecture:**
- `--conditioning_dim`: Conditioning vector dimension (default: 256)
- `--property_hidden_dim`: Hidden dimension for property MLP (default: 512)
- `--property_n_layers`: Number of layers in property MLP (default: 3)
- `--nf`: Hidden feature dimension for EGNN (default: 128)
- `--n_layers`: Number of EGNN layers (default: 6)

**Conditioning Options:**
- `--condition_on_property`: Enable property conditioning (required: True)
- `--condition_on_molecule`: Enable molecular conditioning (recommended: True)
- `--condition_on_space_group`: Enable space group conditioning (optional)
- `--condition_on_density`: Enable density conditioning (optional)

**Training:**
- `--n_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--test_epochs`: Validate every N epochs (default: 10)

**Output:**
- `outputs/{exp_name}/checkpoints/`: Model checkpoints
- `outputs/{exp_name}/checkpoints/best_model.pt`: Best model by validation loss

### Advanced Training Options

**Multiple Properties:**
```bash
--property_names bandgap melting_point dielectric_constant formation_energy
```

**Combined Conditioning:**
```bash
--condition_on_molecule True \
--condition_on_space_group True \
--condition_on_density True \
--condition_on_property True
```

**Resume Training:**
```bash
python main_crystal_with_properties.py \
    --resume outputs/crystal_bandgap/checkpoints/checkpoint_epoch0100.pt \
    [... other arguments ...]
```

### Monitoring Training

The training script logs metrics to Weights & Biases (W&B):
- Training loss
- Validation loss
- Property statistics
- Generated structure metrics

To disable W&B:
```bash
--no_wandb
```

---

## Generation

### Basic Generation Command

Generate crystals with target property values:

```bash
python generate_crystal_with_all_conditions.py \
    --model_path outputs/crystal_bandgap/checkpoints/best_model.pt \
    --molecule_db_path data/molecules.db \
    --molecule_id benzene_001 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --target_dielectric_constant 3.0 \
    --n_samples 100 \
    --output_dir generated_crystals/benzene_target_2.5eV \
    --output_format cif
```

**Key Arguments:**

**Model & Data:**
- `--model_path`: Path to trained model checkpoint
- `--molecule_db_path`: Path to molecule database

**Molecule Selection:**
- `--molecule_id`: Molecule ID (e.g., 'benzene_001')
- OR `--molecule_index`: Database index (e.g., 0)

**Target Conditions:**
- `--target_<property_name>`: Target value for each property
  - Example: `--target_bandgap 2.5`
  - Must provide values for ALL properties model was trained with

**Optional Conditions:**
- `--space_group`: Target space group number (1-230)
- `--density`: Target density (g/cm³)

**Generation Settings:**
- `--n_samples`: Number of crystals to generate (default: 100)
- `--batch_size`: Batch size for generation (default: 10)
- `--num_atoms`: Number of atoms per crystal (default: from molecule)

**Output:**
- `--output_dir`: Directory for generated crystals
- `--output_format`: Format - 'cif', 'xyz', or 'both'

### Output Structure

Generated files in output directory:
```
generated_crystals/benzene_target_2.5eV/
├── crystal_0000.cif
├── crystal_0000_metadata.json
├── crystal_0001.cif
├── crystal_0001_metadata.json
├── ...
└── generation_summary.json
```

**Metadata JSON:**
```json
{
  "crystal_id": "crystal_0000",
  "molecule_id": "benzene_001",
  "space_group": null,
  "density": null,
  "target_properties": {
    "bandgap": 2.5,
    "melting_point": 180.0,
    "dielectric_constant": 3.0
  },
  "generation_time": "2025-10-24T12:34:56.789",
  "num_atoms": 48
}
```

### Advanced Generation

**Batch Generation with Different Targets:**

Create a CSV with multiple target property sets, then iterate:

```bash
# For each row in targets.csv
for TARGET in 2.0 2.5 3.0; do
    python generate_crystal_with_all_conditions.py \
        --model_path outputs/model.pt \
        --molecule_db_path data/molecules.db \
        --molecule_id benzene_001 \
        --target_bandgap $TARGET \
        --target_melting_point 180.0 \
        --n_samples 50 \
        --output_dir generated/bandgap_${TARGET}
done
```

**Generate with All Conditions:**
```bash
python generate_crystal_with_all_conditions.py \
    --model_path outputs/model.pt \
    --molecule_db_path data/molecules.db \
    --molecule_id benzene_001 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --space_group 14 \
    --density 1.2 \
    --n_samples 100 \
    --output_dir generated/fully_conditioned
```

---

## Complete Workflow Example

### End-to-End Example

Here's a complete workflow from raw data to generated crystals:

```bash
#!/bin/bash
# Complete property-conditioned crystal generation workflow

# 1. Prepare data
echo "Step 1: Preparing dataset..."
python scripts/prepare_property_dataset.py \
    --crystal_db data/raw_crystals.db \
    --property_csv data/properties.csv \
    --output_db data/crystals_with_props.db \
    --property_names bandgap melting_point \
    --validate

# 2. Train model
echo "Step 2: Training model..."
python main_crystal_with_properties.py \
    --crystal_db_path data/crystals_with_props.db \
    --molecule_db_path data/molecules.db \
    --property_names bandgap melting_point \
    --exp_name crystal_bg_mp \
    --n_epochs 200 \
    --batch_size 32 \
    --lr 1e-4 \
    --condition_on_property True \
    --condition_on_molecule True

# 3. Generate crystals with target properties
echo "Step 3: Generating crystals..."
python generate_crystal_with_all_conditions.py \
    --model_path outputs/crystal_bg_mp/checkpoints/best_model.pt \
    --molecule_db_path data/molecules.db \
    --molecule_id target_molecule \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --n_samples 100 \
    --output_dir generated/target_2.5eV_180K

echo "Workflow complete!"
echo "Generated crystals in: generated/target_2.5eV_180K/"
```

### Quick Start with Test Data

For testing with minimal setup:

```bash
# 1. Create synthetic test data (if you don't have real data)
python examples/create_synthetic_property_dataset.py \
    --n_crystals 100 \
    --output_dir test_data/

# 2. Train for a few epochs
python main_crystal_with_properties.py \
    --crystal_db_path test_data/crystals.db \
    --molecule_db_path test_data/molecules.db \
    --property_names bandgap melting_point \
    --exp_name quick_test \
    --n_epochs 10 \
    --batch_size 16 \
    --break_train_epoch True  # Only 1 batch per epoch for testing

# 3. Generate a few samples
python generate_crystal_with_all_conditions.py \
    --model_path outputs/quick_test/checkpoints/checkpoint_epoch0010.pt \
    --molecule_db_path test_data/molecules.db \
    --molecule_index 0 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --n_samples 10 \
    --output_dir test_generated/
```

---

## Troubleshooting

### Common Issues

**1. "Property names mismatch" error during generation:**
```
ValueError: Property names mismatch!
```
**Solution:** Ensure you're providing target values for exactly the same properties the model was trained with.

**2. "Missing target value for property" error:**
```
ValueError: Missing target value for property: bandgap
```
**Solution:** Add `--target_bandgap <value>` for each property.

**3. "Properties missing for crystals" error during data prep:**
```
ValueError: Properties missing for 5 crystals
```
**Solution:** Ensure your property CSV has entries for all crystals in the database.

**4. "Zero variance" error:**
```
ValueError: Properties with zero variance: ['property_name']
```
**Solution:** Remove constant-valued properties or add variation to the data.

### Performance Tips

**Training Speed:**
- Use smaller `--batch_size` for large crystals
- Reduce `--n_layers` if memory constrained
- Use `--break_train_epoch True` for quick debugging

**Generation Speed:**
- Increase `--batch_size` for faster generation
- Reduce `--diffusion_steps` in checkpoint args (requires retraining)

**Memory Usage:**
- Reduce `--nf` (hidden dimensions)
- Reduce `--batch_size`
- Use `--max_atoms` to limit crystal size

---

## Next Steps

- See `doc/gen_molecular_crystal/` for detailed specifications
- Check `examples/property_conditioning_example.py` for programmatic usage
- Review `tests/test_integration_property_conditioning.py` for integration examples

---

**Note:** All scripts follow the principle of **no fallback mechanisms**. Invalid inputs will raise explicit errors rather than silently defaulting to fallback behavior.
