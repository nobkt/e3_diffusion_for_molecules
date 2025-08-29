#!/usr/bin/env python3
"""
Example script demonstrating the new molecular descriptor conditioning for ASE database.

This script shows how to:
1. Create a sample ASE database with molecules
2. Train a model with the new molecular descriptor conditioning
3. Generate molecules with specific conditions

Requirements:
- ASE 
- OpenBabel
- E3 Diffusion dependencies
"""

import os
import sys
import tempfile
import numpy as np
from ase import Atoms
from ase.db import connect

def create_sample_ase_database(db_path, n_molecules=50):
    """
    Create a sample ASE database with various molecules for testing
    
    Parameters
    ----------
    db_path : str
        Path where to create the database
    n_molecules : int
        Number of molecules to generate
    """
    
    print(f"Creating sample ASE database at {db_path} with {n_molecules} molecules...")
    
    # Connect to database
    db = connect(db_path)
    
    # Define some simple molecular templates
    molecules = []
    
    # 1. Water molecules with small variations
    for i in range(n_molecules // 5):
        positions = [
            [0, 0, 0], 
            [0.757 + np.random.normal(0, 0.05), 0.586 + np.random.normal(0, 0.05), 0], 
            [-0.757 + np.random.normal(0, 0.05), 0.586 + np.random.normal(0, 0.05), 0]
        ]
        water = Atoms('H2O', positions=positions)
        molecules.append((water, {
            'energy': -76.4 + np.random.normal(0, 0.5),
            'homo': -12.6 + np.random.normal(0, 0.3),
            'lumo': 1.4 + np.random.normal(0, 0.2)
        }))
    
    # 2. Methane molecules with small variations
    for i in range(n_molecules // 5):
        # Basic tetrahedral structure with noise
        positions = [
            [0, 0, 0],  # C center
            [1.089 + np.random.normal(0, 0.05), 1.089 + np.random.normal(0, 0.05), 1.089 + np.random.normal(0, 0.05)],
            [1.089 + np.random.normal(0, 0.05), -1.089 + np.random.normal(0, 0.05), -1.089 + np.random.normal(0, 0.05)],
            [-1.089 + np.random.normal(0, 0.05), 1.089 + np.random.normal(0, 0.05), -1.089 + np.random.normal(0, 0.05)],
            [-1.089 + np.random.normal(0, 0.05), -1.089 + np.random.normal(0, 0.05), 1.089 + np.random.normal(0, 0.05)]
        ]
        methane = Atoms('CH4', positions=positions)
        molecules.append((methane, {
            'energy': -40.5 + np.random.normal(0, 0.3),
            'homo': -14.4 + np.random.normal(0, 0.3),
            'lumo': 6.0 + np.random.normal(0, 0.3)
        }))
    
    # 3. Ammonia molecules
    for i in range(n_molecules // 5):
        positions = [
            [0, 0, 0],  # N center
            [1.017 + np.random.normal(0, 0.05), 0, 0],
            [-0.509 + np.random.normal(0, 0.05), 0.882 + np.random.normal(0, 0.05), 0],
            [-0.509 + np.random.normal(0, 0.05), -0.882 + np.random.normal(0, 0.05), 0]
        ]
        ammonia = Atoms('NH3', positions=positions)
        molecules.append((ammonia, {
            'energy': -56.5 + np.random.normal(0, 0.4),
            'homo': -10.8 + np.random.normal(0, 0.3),
            'lumo': 2.1 + np.random.normal(0, 0.3)
        }))
    
    # 4. Ethane molecules  
    for i in range(n_molecules // 5):
        positions = [
            [0, 0, 0],      # C1
            [1.54, 0, 0],   # C2
            [-0.51, 0.88, 0], [-0.51, -0.44, 0.76], [-0.51, -0.44, -0.76],  # H on C1
            [2.05, 0.88, 0], [2.05, -0.44, 0.76], [2.05, -0.44, -0.76]      # H on C2
        ]
        # Add small random variations
        positions = [[x + np.random.normal(0, 0.05), y + np.random.normal(0, 0.05), z + np.random.normal(0, 0.05)] 
                    for x, y, z in positions]
        ethane = Atoms('C2H6', positions=positions)
        molecules.append((ethane, {
            'energy': -79.8 + np.random.normal(0, 0.4),
            'homo': -13.6 + np.random.normal(0, 0.3),
            'lumo': 5.8 + np.random.normal(0, 0.3)
        }))
    
    # 5. Carbon dioxide molecules
    for i in range(n_molecules - len(molecules)):
        positions = [
            [0, 0, 0],      # C center
            [1.16, 0, 0],   # O1
            [-1.16, 0, 0]   # O2
        ]
        # Add small random variations
        positions = [[x + np.random.normal(0, 0.05), y + np.random.normal(0, 0.05), z + np.random.normal(0, 0.05)] 
                    for x, y, z in positions]
        co2 = Atoms('CO2', positions=positions)
        molecules.append((co2, {
            'energy': -188.6 + np.random.normal(0, 0.5),
            'homo': -13.8 + np.random.normal(0, 0.3),
            'lumo': 4.0 + np.random.normal(0, 0.3)
        }))
    
    # Add molecules to database
    for atoms, properties in molecules:
        db.write(atoms, data=properties)
    
    print(f"Created database with {len(molecules)} molecules")
    print("Molecule types:", {mol[0].get_chemical_formula(): 1 for mol in molecules}.keys())
    return db_path


def demonstrate_conditioning_usage():
    """
    Demonstrate how to use the new molecular descriptor conditioning
    """
    
    print("\n" + "="*60)
    print("MOLECULAR DESCRIPTOR CONDITIONING EXAMPLE")
    print("="*60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create sample database
        create_sample_ase_database(db_path, n_molecules=100)
        
        # Show example command lines for different conditioning scenarios
        
        print("\n" + "="*60)
        print("EXAMPLE USAGE COMMANDS")
        print("="*60)
        
        print("\n1. TRAINING WITH MOLECULAR WEIGHT CONDITIONING:")
        print("-" * 50)
        cmd1 = f"""python main_qm9.py \\
    --dataset ase_db \\
    --ase_db_path {db_path} \\
    --conditioning molecular_weight \\
    --exp_name ase_molecular_weight_cond \\
    --n_epochs 100 \\
    --batch_size 32"""
        print(cmd1)
        
        print("\n2. TRAINING WITH ATOM TYPES CONDITIONING:")
        print("-" * 50)
        cmd2 = f"""python main_qm9.py \\
    --dataset ase_db \\
    --ase_db_path {db_path} \\
    --conditioning atom_types_encoding \\
    --exp_name ase_atom_types_cond \\
    --n_epochs 100 \\
    --batch_size 32"""
        print(cmd2)
        
        print("\n3. TRAINING WITH π CONJUGATION RATIO CONDITIONING:")
        print("-" * 50)
        cmd3 = f"""python main_qm9.py \\
    --dataset ase_db \\
    --ase_db_path {db_path} \\
    --conditioning pi_conjugation_ratio \\
    --exp_name ase_pi_conjugation_cond \\
    --n_epochs 100 \\
    --batch_size 32"""
        print(cmd3)
        
        print("\n4. TRAINING WITH MULTIPLE CONDITIONS:")
        print("-" * 50)
        cmd4 = f"""python main_qm9.py \\
    --dataset ase_db \\
    --ase_db_path {db_path} \\
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding \\
    --exp_name ase_multi_cond \\
    --n_epochs 100 \\
    --batch_size 32"""
        print(cmd4)
        
        print("\n5. TRAINING WITH FUNCTIONAL GROUPS CONDITIONING:")
        print("-" * 50)
        cmd5 = f"""python main_qm9.py \\
    --dataset ase_db \\
    --ase_db_path {db_path} \\
    --conditioning functional_groups_encoding \\
    --exp_name ase_functional_groups_cond \\
    --n_epochs 100 \\
    --batch_size 32"""
        print(cmd5)
        
        print("\n" + "="*60)
        print("EVALUATION COMMANDS")
        print("="*60)
        
        print("\nEVALUATE TRAINED MODEL:")
        print("-" * 30)
        eval_cmd = f"""python eval_analyze.py \\
    --model_path outputs/ase_molecular_weight_cond \\
    --n_samples 1000 \\
    --ase_db_path {db_path}"""
        print(eval_cmd)
        
        print("\nGENERATE SAMPLES:")
        print("-" * 20)
        sample_cmd = f"""python eval_sample.py \\
    --model_path outputs/ase_molecular_weight_cond \\
    --n_tries 10 \\
    --ase_db_path {db_path}"""
        print(sample_cmd)
        
        # Show molecular descriptor information
        print("\n" + "="*60)
        print("EXTRACTED MOLECULAR DESCRIPTORS")
        print("="*60)
        
        # Load and analyze the created database
        from qm9.dataset import load_ase_database
        
        try:
            datasets, num_species, charge_scale = load_ase_database(
                db_path, 
                split_ratios=(0.8, 0.1, 0.1), 
                include_charges=True, 
                remove_h=False
            )
            
            train_data = datasets['train'].data
            print(f"\nDATASET STATISTICS:")
            print(f"Number of molecules: {len(train_data['num_atoms'])}")
            print(f"Available properties: {[k for k in train_data.keys() if not k.startswith('_')]}")
            
            if 'molecular_weight' in train_data:
                mw = train_data['molecular_weight']
                print(f"Molecular weight range: {mw.min():.2f} - {mw.max():.2f} u")
            
            if 'pi_conjugation_ratio' in train_data:
                pi_ratio = train_data['pi_conjugation_ratio']
                print(f"π conjugation ratio range: {pi_ratio.min():.3f} - {pi_ratio.max():.3f}")
            
            if 'atom_types_encoding' in train_data:
                atom_encoding = train_data['atom_types_encoding']
                print(f"Atom types encoding shape: {atom_encoding.shape}")
                if '_atom_types_mapping' in train_data:
                    print(f"Atom types found: {train_data['_atom_types_mapping']}")
            
            if 'functional_groups_encoding' in train_data:
                fg_encoding = train_data['functional_groups_encoding']
                print(f"Functional groups encoding shape: {fg_encoding.shape}")
                if '_functional_groups_mapping' in train_data:
                    print(f"Functional groups found: {train_data['_functional_groups_mapping']}")
                    
        except Exception as e:
            print(f"Warning: Could not analyze database: {e}")
        
        print("\n" + "="*60)
        print("NOTES")
        print("="*60)
        print("""
1. The new conditioning options are:
   - molecular_weight: Condition on the molecular weight in atomic mass units
   - pi_conjugation_ratio: Condition on the ratio of π bonds to total bonds
   - atom_types_encoding: Condition on the presence of specific atom types
   - functional_groups_encoding: Condition on the presence of functional groups

2. These conditions use ASE and OpenBabel for extraction, avoiding RDKit dependency.

3. The atom_types_encoding and functional_groups_encoding are binary vectors
   where each element indicates the presence/absence of a specific type/group.

4. You can combine multiple conditions for more precise control over generation.

5. Make sure your ASE database contains sufficient molecular diversity for 
   effective conditional training.
        """)
        
    finally:
        # Clean up temporary file
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\nCleaned up temporary database: {db_path}")


if __name__ == "__main__":
    try:
        demonstrate_conditioning_usage()
        print("\n🎉 Example demonstration completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)