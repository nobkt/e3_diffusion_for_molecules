# ASE Database Validation

This document describes the ASE database validation functionality that helps identify potential data quality issues before training.

## Problem

When using `dataset=ase`, users may encounter very high loss values (25-28) and fragmented molecules after sampling. These issues are often caused by problematic data in the ASE database file.

## Solution

A new validation option `--validate_ase_db` has been added to check the quality of ASE database files before training.

## Usage

### Option 1: Validate during training

Add `--validate_ase_db` to your training command:

```bash
python main_qm9.py --dataset ase --ase_db_file your_database.db --validate_ase_db [other training options]
```

If validation fails, training will stop with an error message indicating the issues found.

### Option 2: Standalone validation

Validate a database file independently:

```bash
python build_ase_dataset.py --db_file your_database.db --validate
```

## What is Validated

The validation checks for:

### 1. Basic Database Issues
- File exists and can be opened
- Database is not empty
- Entries can be read successfully

### 2. Molecular Structure Quality
- **Bond lengths**: Checks for unreasonably short bonds (< 0.5 Å) that indicate overlapping atoms
- **Large distances**: Warns about very long distances (> 20 Å) that may indicate molecular fragments
- **Element types**: Warns about uncommon elements that may not be compatible with the model

### 3. Property Value Ranges
- **Energy values**: Checks for extremely large energy values that may indicate unit errors
- **Property availability**: Verifies that expected molecular properties are present

### 4. Statistical Summary
- Element distribution
- Molecule size range and average
- Bond length statistics
- Property value ranges

## Example Output

### Successful Validation
```
======================================================================
Validating ASE Database: molecules.db
======================================================================
✓ Database file exists: molecules.db
✓ Successfully connected to database
✓ Database contains 1000 molecules

📊 Database Statistics:
   Examined: 1000 molecules
   Valid structures: 1000
   Elements found: [1, 6, 7, 8, 9]
   Molecule sizes: 3-29 atoms (avg: 15.2)
   Properties: ['HOMO_Ha', 'LUMO_Ha', 'U0_Ha', 'alpha', 'gap_Ha']

======================================================================
✅ VALIDATION PASSED
Database appears suitable for training.
======================================================================
```

### Failed Validation
```
======================================================================
Validating ASE Database: problematic.db
======================================================================
✓ Database file exists: problematic.db
✓ Successfully connected to database
✓ Database contains 100 molecules

📊 Database Statistics:
   Examined: 100 molecules
   Valid structures: 85
   Elements found: [1, 6, 7, 8]
   Molecule sizes: 2-25 atoms (avg: 12.1)
   Properties: ['U0_Ha', 'alpha']

⚠️  Warnings (5):
   Molecule 12: Unusually large energy -50000.0 Ha
   Molecule 23: Very long distance 25.3 Å (possible fragment)
   ...

❌ Issues (15):
   Molecule 3: Unreasonably short bond distance 0.1 Å
   Molecule 15: Unreasonably short bond distance 0.2 Å
   ...

======================================================================
❌ VALIDATION FAILED
Database has issues that may cause training problems:
  • 15 critical structural issues
  • High issue rate: 15 issues out of 85 molecules

🔧 Recommendations:
  • Check molecular structures for reasonable bond lengths
  • Verify energy values are in expected ranges
  • Consider filtering or cleaning the database
======================================================================
```

## Common Issues and Solutions

### 1. Short Bond Lengths
**Problem**: Atoms are too close together (< 0.5 Å)
**Causes**: 
- Overlapping atoms in molecular structure
- Unit conversion errors (e.g., Bohr vs Angstrom)
- Optimization artifacts

**Solutions**:
- Re-optimize molecular geometries
- Check coordinate units
- Filter out problematic structures

### 2. Large Energy Values
**Problem**: Energy values are unusually large
**Causes**:
- Wrong units (e.g., Joules instead of Hartree)
- Calculation errors
- Unconverged optimizations

**Solutions**:
- Verify energy units (should be in Hartree for *_Ha properties)
- Check calculation convergence
- Apply reasonable energy filters

### 3. Molecular Fragments
**Problem**: Very large distances between atoms
**Causes**:
- Dissociated molecules
- Multi-molecular systems in single entry
- Calculation artifacts

**Solutions**:
- Use only single, intact molecules
- Filter by maximum molecular span
- Check molecular connectivity

## Integration with Training

When validation is enabled with `--validate_ase_db`, the training script will:

1. Run validation before loading the dataset
2. Stop training if critical issues are found
3. Continue training if validation passes
4. Print summary statistics about the database

This prevents wasting computation time on training with problematic data that would result in poor model performance.

## Customization

The validation parameters can be customized by modifying the `validate_ase_database()` function in `build_ase_dataset.py`:

- `max_entries`: Limit validation to first N molecules (for large databases)
- Bond length thresholds (currently 0.5 Å minimum, 20 Å maximum)
- Energy value thresholds (database-dependent)
- Element type restrictions

## Performance Notes

- Validation time scales with database size
- For large databases (>10,000 molecules), consider using `--max_entries` to limit validation scope
- Validation is much faster than training, so the overhead is minimal

## Example Workflow

```bash
# 1. First validate your database
python build_ase_dataset.py --db_file my_molecules.db --validate

# 2. If validation passes, start training with validation enabled
python main_qm9.py --dataset ase --ase_db_file my_molecules.db --validate_ase_db \
  --n_epochs 1000 --exp_name ase_experiment --batch_size 32

# 3. If validation fails, fix the database and repeat
```

This ensures that data quality issues are caught early, leading to more successful training runs.