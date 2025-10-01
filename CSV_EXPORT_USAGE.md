# CSV Export Feature for Generation Conditions

## Overview

This feature allows you to export generation conditions (molecular descriptors) for all molecules in a dataset to CSV files. The exported conditions include:

1. **molecular_weight** - Molecular weight in Daltons (Da)
2. **pi_conjugation_ratio** - π-conjugation ratio (0.0 to 1.0)
3. **atom_types_encoding** - One-hot encoding of atom types present in the molecule
4. **functional_groups_encoding** - One-hot encoding of functional groups present in the molecule

## Usage

### Command Line

Use the `--export_conditions_csv` argument to specify the output directory for CSV files:

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path path/to/your/database.db \
    --export_conditions_csv output_directory \
    --no_wandb \
    --remove_h
```

### Arguments

- `--export_conditions_csv <directory>`: Output directory for CSV files. If not specified, the feature is disabled and normal training proceeds.
- `--dataset`: Dataset type (`ase_db` for ASE database, `qm9` for QM9 dataset)
- `--ase_db_path`: Path to ASE database file (required when using `ase_db` dataset)
- `--remove_h`: Remove hydrogen atoms (optional, recommended for ASE databases)
- `--no_wandb`: Disable Weights & Biases logging (recommended for CSV export mode)

### Output Files

The feature generates four CSV files in the specified output directory:

#### 1. molecular_weight.csv

Contains molecular weight for each molecule.

**Format:**
```csv
ID,分子の組成,分子量
0,C6H12O6,180.1559
1,C8H10N4O2,194.1906
...
```

**Columns:**
- `ID`: Molecule ID (index in dataset)
- `分子の組成`: Molecular composition/formula (e.g., C6H12O6)
- `分子量`: Molecular weight in Daltons

#### 2. pi_conjugation_ratio.csv

Contains π-conjugation ratio for each molecule.

**Format:**
```csv
ID,分子の組成,π共役比率
0,C6H12O6,0.1667
1,C8H10N4O2,0.5556
...
```

**Columns:**
- `ID`: Molecule ID
- `分子の組成`: Molecular composition/formula
- `π共役比率`: π-conjugation ratio (0.0 to 1.0)

#### 3. atom_types_encoding.csv

Contains one-hot encoding of atom types for each molecule.

**Format:**
```csv
ID,分子の組成,C,H,N,O,F
0,C6H12O6,1,1,0,1,0
1,C8H10N4O2,1,1,1,1,0
...
```

**Columns:**
- `ID`: Molecule ID
- `分子の組成`: Molecular composition/formula
- Additional columns for each atom type (e.g., C, H, N, O, F)
  - Value 1 indicates the atom type is present in the molecule
  - Value 0 indicates the atom type is not present

#### 4. functional_groups_encoding.csv

Contains one-hot encoding of functional groups for each molecule.

**Format:**
```csv
ID,分子の組成,carbonyl,hydroxyl,amine,carboxyl
0,C6H12O6,1,1,0,1
1,C8H10N4O2,1,0,1,0
...
```

**Columns:**
- `ID`: Molecule ID
- `分子の組成`: Molecular composition/formula
- Additional columns for each functional group
  - Value 1 indicates the functional group is present
  - Value 0 indicates the functional group is not present

## Examples

### Example 1: Export from ASE Database

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path ase.db \
    --export_conditions_csv ./csv_output \
    --no_wandb \
    --remove_h
```

This will:
1. Load molecules from `ase.db`
2. Compute generation conditions for all molecules
3. Export CSV files to `./csv_output/` directory
4. Exit the program

### Example 2: Export from QM9 Dataset

```bash
python main_qm9.py \
    --dataset qm9 \
    --datadir qm9/temp \
    --export_conditions_csv ./qm9_conditions \
    --no_wandb
```

This will:
1. Load QM9 dataset
2. Compute generation conditions (where available)
3. Export CSV files to `./qm9_conditions/` directory
4. Exit the program

## Notes

1. **Program Exit**: When `--export_conditions_csv` is specified, the program will exit after generating the CSV files. No training or other operations will be performed.

2. **Data Availability**: 
   - For ASE databases, molecular descriptors are computed using OpenBabel
   - For QM9 dataset, some descriptors may not be available (will be noted in the output)

3. **Molecular Composition**: 
   - Composition strings show the molecular formula (e.g., C6H12O6)
   - Elements are sorted alphabetically in the formula
   - If composition cannot be determined, "Unknown" or "Molecule_<ID>" is used

4. **Encoding Format**:
   - Atom types and functional groups use binary encoding (0 or 1)
   - Column headers show the specific atom types or functional groups
   - The set of columns depends on what's present in the dataset

## Requirements

- The feature works with both ASE database and QM9 datasets
- For ASE databases with molecular descriptors, OpenBabel must be installed
- Unicode CSV files are generated (UTF-8 encoding) to support Japanese headers

## Troubleshooting

**Issue**: No CSV files generated
- **Solution**: Check that the output directory is writable and the dataset loads successfully

**Issue**: Missing descriptors in output
- **Solution**: For ASE databases, ensure OpenBabel is installed: `pip install openbabel-wheel`

**Issue**: Empty CSV files
- **Solution**: Verify the dataset contains molecules and descriptors are computed correctly
