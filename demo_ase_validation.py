#!/usr/bin/env python3
"""
Demonstration script for ASE database validation functionality.

This script shows how to use the new validation feature to check ASE databases
before training, helping to prevent high loss values and fragmented molecules.
"""

import os
import ase
import ase.db
import tempfile
import subprocess

def create_demo_databases():
    """Create demonstration databases: one good, one problematic."""
    
    # Good database
    good_db_path = '/tmp/good_molecules.db'
    with ase.db.connect(good_db_path) as db:
        # Water molecule - good structure
        atoms = ase.Atoms('OH2', positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        db.write(atoms, U0_Ha=-76.4, HOMO_Ha=-0.59, LUMO_Ha=0.08, gap_Ha=0.67, alpha=9.5)
        
        # Methane molecule - good structure
        atoms = ase.Atoms('CH4', positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]])
        db.write(atoms, U0_Ha=-40.5, HOMO_Ha=-0.60, LUMO_Ha=0.06, gap_Ha=0.66, alpha=17.3)
        
        # Ammonia molecule - good structure
        atoms = ase.Atoms('NH3', positions=[[0, 0, 0], [0.9, 0.3, 0.3], [-0.3, 0.9, 0.3], [-0.3, -0.3, 0.9]])
        db.write(atoms, U0_Ha=-56.5, HOMO_Ha=-0.49, LUMO_Ha=0.03, gap_Ha=0.52, alpha=14.8)
    
    # Problematic database
    bad_db_path = '/tmp/bad_molecules.db'
    with ase.db.connect(bad_db_path) as db:
        # Molecule with atoms too close together
        atoms = ase.Atoms('HH', positions=[[0, 0, 0], [0.1, 0, 0]])  # 0.1 Å - too close!
        db.write(atoms, U0_Ha=-1.0, alpha=2.5)
        
        # Molecule with unreasonable energy values
        atoms = ase.Atoms('CH4', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-0.5, -0.5, -0.5]])
        db.write(atoms, U0_Ha=-50000.0, alpha=17.3)  # Energy too large!
        
        # Molecule with very spread out atoms (possible fragment)
        atoms = ase.Atoms('CO', positions=[[0, 0, 0], [25, 0, 0]])  # 25 Å apart - likely fragment
        db.write(atoms, U0_Ha=-113.3, alpha=13.1)
    
    return good_db_path, bad_db_path

def run_validation_demo():
    """Run the validation demonstration."""
    
    print("=" * 80)
    print("ASE Database Validation Demo")
    print("=" * 80)
    print("This demo shows how to validate ASE databases before training")
    print("to prevent high loss values and fragmented molecules.\n")
    
    # Create demo databases
    print("Creating demonstration databases...")
    good_db, bad_db = create_demo_databases()
    print(f"✓ Good database: {good_db}")
    print(f"✓ Bad database: {bad_db}")
    
    print("\n" + "=" * 80)
    print("1. VALIDATING GOOD DATABASE")
    print("=" * 80)
    print("This database should pass validation:")
    print(f"Command: python build_ase_dataset.py --db_file {good_db} --validate\n")
    
    # Run validation on good database
    result = subprocess.run(['python', 'build_ase_dataset.py', '--db_file', good_db, '--validate'], 
                          capture_output=False)
    print(f"\nGood database validation exit code: {result.returncode} (0 = passed)")
    
    print("\n" + "=" * 80)
    print("2. VALIDATING PROBLEMATIC DATABASE")
    print("=" * 80)
    print("This database should fail validation due to structural issues:")
    print(f"Command: python build_ase_dataset.py --db_file {bad_db} --validate\n")
    
    # Run validation on bad database
    result = subprocess.run(['python', 'build_ase_dataset.py', '--db_file', bad_db, '--validate'], 
                          capture_output=False)
    print(f"\nBad database validation exit code: {result.returncode} (1 = failed)")
    
    print("\n" + "=" * 80)
    print("3. USAGE IN TRAINING")
    print("=" * 80)
    print("To use validation in training, add --validate_ase_db to your command:")
    print()
    print("✅ GOOD DATABASE - Training would proceed:")
    print(f"python main_qm9.py --dataset ase --ase_db_file {good_db} --validate_ase_db [other options]")
    print()
    print("❌ BAD DATABASE - Training would stop with error:")
    print(f"python main_qm9.py --dataset ase --ase_db_file {bad_db} --validate_ase_db [other options]")
    
    print("\n" + "=" * 80)
    print("4. COMMON VALIDATION ISSUES AND SOLUTIONS")
    print("=" * 80)
    print("""
Common Issues Found by Validation:

1. SHORT BOND LENGTHS (< 0.5 Å)
   Problem: Atoms are overlapping
   Solutions: 
   - Re-optimize molecular geometries
   - Check coordinate units (Bohr vs Angstrom)
   - Remove problematic structures

2. LARGE ENERGY VALUES 
   Problem: Energy values are unreasonably large
   Solutions:
   - Verify energy units (should be Hartree for *_Ha properties)
   - Check calculation convergence
   - Apply energy filters

3. MOLECULAR FRAGMENTS
   Problem: Very large distances between atoms (> 20 Å)
   Solutions:
   - Use only intact, single molecules
   - Filter by molecular span
   - Check molecular connectivity

4. MISSING PROPERTIES
   Warning: Expected properties not found
   Solutions:
   - Add required molecular properties
   - Ensure consistent property naming
   - Check property extraction scripts
""")
    
    print("\n" + "=" * 80)
    print("5. BEST PRACTICES")
    print("=" * 80)
    print("""
Recommended Workflow:

1. Always validate your ASE database first:
   python build_ase_dataset.py --db_file your_data.db --validate

2. Fix any issues found by validation

3. Run training with validation enabled:
   python main_qm9.py --dataset ase --ase_db_file your_data.db --validate_ase_db [options]

4. For large databases, use --max_entries for faster validation:
   python build_ase_dataset.py --db_file large_db.db --validate --max_entries 1000

This workflow helps ensure successful training with stable molecular generation.
""")
    
    # Cleanup
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)
    if os.path.exists(good_db):
        os.remove(good_db)
        print(f"✓ Removed {good_db}")
    if os.path.exists(bad_db):
        os.remove(bad_db)
        print(f"✓ Removed {bad_db}")
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETED")
    print("=" * 80)
    print("You can now use ASE database validation to improve your training results!")

if __name__ == "__main__":
    run_validation_demo()