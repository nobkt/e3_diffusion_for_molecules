# General Molecular Database Support for E3 Diffusion

This repository supports loading molecular datasets from various sources including PubChem and other general molecular databases through ASE (Atomic Simulation Environment) database format, in addition to the original QM9 and GEOM datasets.

## Key Features

### 🚀 **NEW: Automatic Configuration Generation**
The system now automatically analyzes any molecular database and creates optimized configurations, supporting:
- **Any combination of elements** from the periodic table
- **Dynamic atom type detection** based on actual database content  
- **Automatic optimization** of model parameters based on dataset characteristics
- **Comprehensive validation** and recommendations for training

### 🧪 **General Molecular Database Support**
- Support for **PubChem** and other large molecular databases
- **Automatic element mapping** for the entire periodic table
- **Dynamic configuration generation** based on database analysis
- **Flexible molecular size handling** (from small molecules to large drug-like compounds)
- **Comprehensive property support** for any molecular descriptors

### 🔧 **Easy-to-Use Utilities**
- **Database analysis** and validation tools
- **Automatic training parameter recommendations**
- **Configuration file generation** for complex databases
- **PubChem SDF conversion** utilities

## Installation

Make sure you have ASE and OpenBabel installed:
```bash
pip install ase openbabel-wheel
```

These are already included in the updated `requirements.txt`.

## Quick Start for General Molecular Databases

### 1. Analyze Your Database

Before training, analyze your molecular database to understand its characteristics:

```bash
python molecular_db_utils.py analyze --db_path your_molecules.db
```

This will show you:
- Number of molecules and elements
- Molecular size distribution  
- Available properties
- Training recommendations

### 2. Automatic Training

For most databases, you can start training immediately without manual configuration:

```bash
python main_qm9.py --dataset ase_db --ase_db_path your_molecules.db
```

The system will automatically:
- Analyze your database
- Create optimal configurations
- Detect all elements present
- Set appropriate model parameters

### 3. Get Training Recommendations

For optimal results, get customized training parameters:

```bash
python molecular_db_utils.py recommend --db_path your_molecules.db
```

### 4. Convert PubChem Data

Convert PubChem SDF files to ASE database format:

```bash
python molecular_db_utils.py convert-pubchem --sdf_path pubchem_data.sdf --output molecules.db
```

## Advanced Usage

### Manual Configuration Creation

For advanced use cases, create custom configurations:

```bash
python molecular_db_utils.py create-config --db_path molecules.db --output custom_config.json
```

Then use the custom configuration:

```bash
python main_qm9.py --dataset custom_config.json
```

### Database Validation

Validate that your database is suitable for training:

```bash
python molecular_db_utils.py validate --db_path molecules.db
```

### Working with Large Databases

For very large databases (>100K molecules), consider:

```bash
# Use larger batch sizes and fewer epochs
python main_qm9.py --dataset ase_db --ase_db_path large_db.db --batch_size 256 --n_epochs 50

# Filter to smaller molecules for faster training
python main_qm9.py --dataset ase_db --ase_db_path large_db.db --filter_n_atoms 30
```

## Usage

### Basic Usage

To train a model using an ASE database, use the `--dataset ase_db` flag along with the path to your database:

```bash
python main_qm9.py --dataset ase_db --ase_db_path /path/to/your/database.db
```

### Command Line Arguments

New arguments for ASE database support:

- `--dataset ase_db`: Use ASE database format
- `--ase_db_path`: Path to the ASE database file (required for ase_db dataset)
- `--split_ratios`: Train, validation, test split ratios as three numbers (default: 0.8 0.1 0.1)
- `--ase_to_eV`: JSON string for unit conversion factors (e.g., `'{"energy": 27.2114}'`)

### New Molecular Descriptor Conditioning

**NEW FEATURE**: When using `dataset=ase_db`, you can now condition molecule generation on molecular descriptors:

#### Available Conditioning Options

1. **Molecular Weight** (`molecular_weight`)
   - Condition on the molecular weight in atomic mass units (u)
   - Extracted using ASE atomic masses

2. **π Conjugation Ratio** (`pi_conjugation_ratio`) 
   - Condition on the ratio of π bonds (double + aromatic) to total bonds
   - Extracted using OpenBabel bond perception

3. **Atom Types** (`atom_types_encoding`)
   - Condition on the presence/absence of specific atom types
   - Binary encoding where each element indicates if an atom type is present
   - Extracted directly from ASE Atoms objects

4. **Functional Groups** (`functional_groups_encoding`)
   - Condition on the presence/absence of functional groups
   - Binary encoding for detected functional groups
   - Extracted using OpenBabel SMARTS pattern matching
   - Detects: hydroxyl (-OH), carbonyl (C=O), carboxyl (-COOH), aldehyde (-CHO), ketone, amino (-NH2), nitro (-NO2), halogens (Cl, Br, F, I), methyl (-CH3), methoxy (-OCH3), phenyl (benzene ring)

### Examples

1. **Basic usage:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db
```

2. **Condition on molecular weight:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --conditioning molecular_weight
```

3. **Condition on π conjugation ratio:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --conditioning pi_conjugation_ratio
```

4. **Condition on atom types:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --conditioning atom_types_encoding
```

5. **Condition on functional groups:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --conditioning functional_groups_encoding
```

6. **Multiple conditions:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding
```

7. **Custom split ratios:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --split_ratios 0.7 0.2 0.1
```

8. **Remove hydrogen atoms:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --remove_h
```

9. **With unit conversion:**
```bash
python main_qm9.py --dataset ase_db --ase_db_path molecules.db --ase_to_eV '{"energy": 27.2114, "homo": 27.2114, "lumo": 27.2114}'
```

### Creating an ASE Database

You can create an ASE database from molecular structures:

```python
from ase import Atoms
from ase.db import connect

# Create database
db = connect('molecules.db')

# Add molecules with properties
water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
db.write(water, data={'energy': -76.4, 'homo': -12.6, 'lumo': 1.4})

methane = Atoms('CH4', positions=[
    [0, 0, 0],        # C
    [1.089, 1.089, 1.089],   # H
    [1.089, -1.089, -1.089], # H
    [-1.089, 1.089, -1.089], # H
    [-1.089, -1.089, 1.089]  # H
])
db.write(methane, data={'energy': -40.5, 'homo': -14.4, 'lumo': 6.0})
```

### Supported Properties

The ASE loader automatically extracts molecular properties from the database. Common properties include:

- `energy`: Total energy
- `homo`: HOMO energy
- `lumo`: LUMO energy
- `gap`: HOMO-LUMO gap
- `mu`: Dipole moment
- `alpha`: Polarizability
- `zpve`: Zero-point vibrational energy
- `U0`, `U`, `H`, `G`, `Cv`: Thermodynamic properties

Any properties stored in the ASE database's `data` field will be automatically included.

**NEW**: Molecular descriptors are automatically computed and added:
- `molecular_weight`: Molecular weight in atomic mass units
- `pi_conjugation_ratio`: Ratio of π bonds to total bonds
- `atom_types_encoding`: Binary encoding of present atom types
- `functional_groups_encoding`: Binary encoding of present functional groups

### Features

- **Automatic atom type detection**: The loader automatically determines all atom types present in your dataset
- **Dynamic configuration**: Dataset parameters are updated based on actual data
- **Flexible splits**: Configurable train/validation/test ratios
- **Unit conversion**: Support for converting units (e.g., Hartree to eV)
- **Hydrogen handling**: Option to include or exclude hydrogen atoms
- **Molecular descriptor extraction**: Automatic computation of molecular descriptors using ASE and OpenBabel
- **No RDKit dependency**: Uses ASE and OpenBabel for molecular analysis
- **Full compatibility**: Works with all existing E3 diffusion model features

### Data Format

The ASE loader converts molecular structures to the same tensor format used by QM9:

- `positions`: Atomic coordinates (Å)
- `charges`: Atomic numbers
- `num_atoms`: Number of atoms per molecule
- Properties are stored as 1D tensors
- Molecular descriptors are stored as additional properties

The data is automatically padded and collated for efficient batch processing.

## Evaluation

After training a model with ASE database, you can use the evaluation scripts:

### eval_analyze.py

Analyze molecular properties, stability, and uniqueness:

```bash
# For a model trained with ASE database
python eval_analyze.py --model_path outputs/my_ase_model --n_samples 100

# If the ASE database path has changed or you want to use a different one
python eval_analyze.py --model_path outputs/my_ase_model --n_samples 100 --ase_db_path /new/path/to/molecules.db
```

### eval_sample.py

Generate molecular samples and visualizations:

```bash
# For a model trained with ASE database
python eval_sample.py --model_path outputs/my_ase_model --n_tries 10 --n_nodes 15

# With custom ASE database path
python eval_sample.py --model_path outputs/my_ase_model --ase_db_path /path/to/molecules.db
```

The evaluation scripts automatically detect when a model was trained with `dataset=ase_db` and load the appropriate ASE database. If the original database path is not available, you can specify a new one using the `--ase_db_path` argument.

## Example Script

Run `python example_ase_conditioning.py` to see a complete demonstration of the molecular descriptor conditioning functionality, including:

- Creating a sample ASE database
- Example training commands for different conditioning scenarios
- Evaluation commands
- Analysis of extracted molecular descriptors

## Implementation Details

### Molecular Descriptor Extraction

The molecular descriptors are extracted using:

1. **ASE (Atomic Simulation Environment)**: For atom types and molecular weight calculation
2. **OpenBabel**: For functional group detection and π conjugation analysis

This approach avoids dependency on RDKit while providing robust molecular analysis capabilities.

### Functional Group Detection

Functional groups are detected using SMARTS pattern matching with the following patterns:

- Hydroxyl: `[OH]`
- Carbonyl: `[CX3]=[OX1]` 
- Carboxyl: `[CX3](=O)[OX2H1]`
- Aldehyde: `[CX3H1](=O)[#6]`
- Ketone: `[CX3](=O)([#6])[#6]`
- Amino: `[NX3;H2,H1;!$(NC=O)]`
- Nitro: `[N+](=O)[O-]`
- Halogens: `[Cl]`, `[Br]`, `[F]`, `[I]`
- Methyl: `[CH3]`
- Methoxy: `[OX2]([#6])[CH3]`
- Phenyl: `c1ccccc1`

### π Conjugation Calculation

The π conjugation ratio is calculated as:
```
π_ratio = (number_of_double_bonds + number_of_aromatic_bonds) / total_number_of_bonds
```

This provides a measure of the degree of conjugation in the molecular system.