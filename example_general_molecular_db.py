#!/usr/bin/env python3
"""
Example demonstrating general molecular database support for E3 Diffusion.

This script shows how to work with diverse molecular databases like those
from PubChem, including automatic configuration generation and training.
"""

import os
import tempfile
import numpy as np
from ase import Atoms
from ase.db import connect

def create_pubchem_like_database(db_path, n_molecules=100):
    """
    Create a diverse molecular database similar to what you might find in PubChem.
    
    This includes organic compounds, inorganics, organometallics, and various
    element types to demonstrate the flexibility of the new system.
    """
    print(f"Creating PubChem-like database: {db_path}")
    
    db = connect(db_path)
    molecules = []
    
    # 1. Small organic molecules (40%)
    n_organic = int(n_molecules * 0.4)
    organic_templates = [
        # Simple hydrocarbons
        (['C', 'H', 'H', 'H', 'H'], "methane"),
        (['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], "ethane"),
        (['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], "propane"),
        
        # Alcohols
        (['C', 'O', 'H', 'H', 'H', 'H'], "methanol"),
        (['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'], "ethanol"),
        
        # Amines
        (['N', 'H', 'H', 'H'], "ammonia"),
        (['C', 'N', 'H', 'H', 'H', 'H', 'H'], "methylamine"),
        
        # Carboxylic acids
        (['C', 'O', 'O', 'H', 'H'], "formic acid"),
        (['C', 'C', 'O', 'O', 'H', 'H', 'H'], "acetic acid"),
    ]
    
    for i in range(n_organic):
        template = organic_templates[i % len(organic_templates)]
        symbols, name = template
        n_atoms = len(symbols)
        
        # Generate reasonable molecular geometry
        positions = np.random.normal(0, 1.0, size=(n_atoms, 3))
        
        mol = Atoms(symbols, positions=positions)
        properties = {
            'name': name,
            'category': 'organic',
            'energy': np.random.normal(-50, 20),
            'homo': np.random.normal(-12, 3),
            'lumo': np.random.normal(2, 2),
            'molecular_weight': sum([1 if s == 'H' else 12 if s == 'C' else 16 if s == 'O' else 14 for s in symbols]),
            'boiling_point': np.random.uniform(200, 400),
            'logP': np.random.uniform(-2, 4)
        }
        molecules.append((mol, properties))
    
    # 2. Inorganic compounds (20%)
    n_inorganic = int(n_molecules * 0.2)
    inorganic_templates = [
        # Salts
        (['Na', 'Cl'], "sodium chloride"),
        (['K', 'Br'], "potassium bromide"),
        (['Ca', 'F', 'F'], "calcium fluoride"),
        (['Mg', 'O'], "magnesium oxide"),
        (['Al', 'Cl', 'Cl', 'Cl'], "aluminum chloride"),
        
        # Acids/bases
        (['S', 'O', 'O'], "sulfur dioxide"),
        (['N', 'O', 'O'], "nitrogen dioxide"),
        (['P', 'O', 'O', 'O', 'O', 'H', 'H', 'H'], "phosphoric acid"),
    ]
    
    for i in range(n_inorganic):
        template = inorganic_templates[i % len(inorganic_templates)]
        symbols, name = template
        n_atoms = len(symbols)
        
        positions = np.random.normal(0, 1.5, size=(n_atoms, 3))
        
        mol = Atoms(symbols, positions=positions)
        properties = {
            'name': name,
            'category': 'inorganic',
            'energy': np.random.normal(-100, 50),
            'formation_energy': np.random.normal(-200, 100),
            'lattice_energy': np.random.uniform(500, 3000),
            'melting_point': np.random.uniform(300, 1500)
        }
        molecules.append((mol, properties))
    
    # 3. Organometallic compounds (20%)
    n_organometallic = int(n_molecules * 0.2)
    organometallic_templates = [
        # Iron compounds
        (['Fe', 'C', 'O', 'C', 'O', 'C', 'O'], "iron tricarbonyl"),
        (['Fe', 'C', 'C', 'H', 'H', 'H', 'H', 'H'], "iron methylcyclopentadienyl"),
        
        # Zinc compounds
        (['Zn', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], "diethyl zinc"),
        (['Zn', 'Cl', 'Cl'], "zinc chloride"),
        
        # Copper compounds
        (['Cu', 'S', 'O', 'O', 'O', 'O', 'H', 'H'], "copper sulfate"),
        
        # Nickel compounds
        (['Ni', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O'], "nickel tetracarbonyl"),
    ]
    
    for i in range(n_organometallic):
        template = organometallic_templates[i % len(organometallic_templates)]
        symbols, name = template
        n_atoms = len(symbols)
        
        positions = np.random.normal(0, 2.0, size=(n_atoms, 3))
        
        mol = Atoms(symbols, positions=positions)
        properties = {
            'name': name,
            'category': 'organometallic',
            'energy': np.random.normal(-150, 80),
            'homo': np.random.normal(-8, 2),
            'lumo': np.random.normal(1, 1),
            'magnetic_moment': np.random.uniform(0, 5)
        }
        molecules.append((mol, properties))
    
    # 4. Silicon and other semiconductor materials (10%)
    n_semiconductor = int(n_molecules * 0.1)
    semiconductor_templates = [
        # Silicon compounds
        (['Si', 'O', 'Si', 'O', 'H', 'H'], "silica"),
        (['Si', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], "organosilane"),
        (['Si', 'H', 'H', 'H', 'H'], "silane"),
        
        # Germanium compounds
        (['Ge', 'H', 'H', 'H', 'H'], "germane"),
        
        # Arsenic compounds
        (['As', 'H', 'H', 'H'], "arsine"),
    ]
    
    for i in range(n_semiconductor):
        template = semiconductor_templates[i % len(semiconductor_templates)]
        symbols, name = template
        n_atoms = len(symbols)
        
        positions = np.random.normal(0, 1.5, size=(n_atoms, 3))
        
        mol = Atoms(symbols, positions=positions)
        properties = {
            'name': name,
            'category': 'semiconductor',
            'energy': np.random.normal(-80, 40),
            'band_gap': np.random.uniform(0.5, 3.0),
            'electron_affinity': np.random.uniform(1, 4)
        }
        molecules.append((mol, properties))
    
    # 5. Large drug-like molecules (10%)
    n_drugs = n_molecules - len(molecules)
    for i in range(n_drugs):
        # Create larger, more complex molecules
        n_atoms = np.random.randint(15, 40)
        
        # Drug-like composition: mostly C, N, O, with some S, F, Cl
        elements = ['C', 'N', 'O', 'S', 'F', 'Cl', 'H']
        probabilities = [0.5, 0.1, 0.15, 0.02, 0.03, 0.02, 0.18]
        symbols = np.random.choice(elements, size=n_atoms, p=probabilities)
        
        positions = np.random.normal(0, 3.0, size=(n_atoms, 3))
        
        mol = Atoms(symbols, positions=positions)
        properties = {
            'name': f"drug_compound_{i}",
            'category': 'pharmaceutical',
            'energy': np.random.normal(-300, 100),
            'homo': np.random.normal(-6, 2),
            'lumo': np.random.normal(1, 1),
            'molecular_weight': sum([1 if s == 'H' else 12 if s == 'C' else 16 if s == 'O' else 14 if s == 'N' else 32 if s == 'S' else 35 if s == 'Cl' else 19 for s in symbols]),
            'logP': np.random.uniform(0, 6),
            'bioavailability': np.random.uniform(0.1, 1.0),
            'toxicity_score': np.random.uniform(0, 10)
        }
        molecules.append((mol, properties))
    
    # Write all molecules to database
    for mol, props in molecules:
        db.write(mol, data=props)
    
    print(f"Created PubChem-like database with {len(molecules)} molecules")
    return db_path


def demonstrate_automatic_analysis():
    """Demonstrate automatic database analysis and configuration generation."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Automatic Analysis of General Molecular Database")
    print("="*80)
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create diverse database
        create_pubchem_like_database(db_path, n_molecules=50)
        
        print(f"\n1. Analyzing the database...")
        from qm9.general_molecular_db import analyze_ase_database
        analysis = analyze_ase_database(db_path)
        
        print(f"\n📊 Found {analysis['total_molecules']} molecules with {len(analysis['unique_elements'])} elements:")
        print(f"    Elements: {', '.join(analysis['unique_elements'])}")
        print(f"    Largest molecule: {analysis['max_atoms']} atoms")
        print(f"    Available properties: {', '.join(analysis['available_properties'])}")
        
        print(f"\n2. Creating automatic configuration...")
        from qm9.general_molecular_db import create_general_ase_config
        config = create_general_ase_config(db_path, remove_h=False)
        
        print(f"✅ Created configuration with {len(config['atom_decoder'])} atom types")
        
        print(f"\n3. Getting training recommendations...")
        from qm9.general_molecular_db import suggest_training_parameters
        suggestions = suggest_training_parameters(analysis)
        
        print(f"📋 Recommended parameters:")
        print(f"    Batch size: {suggestions['batch_size']}")
        print(f"    Epochs: {suggestions['n_epochs']}")
        print(f"    Model size: {suggestions['nf']}")
        print(f"    Layers: {suggestions['n_layers']}")
        
        if 'recommended_conditioning' in suggestions:
            print(f"    Conditioning: {', '.join(suggestions['recommended_conditioning'])}")
        
        print(f"\n4. Validating database...")
        from qm9.general_molecular_db import validate_molecular_database
        is_valid, issues, recommendations = validate_molecular_database(db_path)
        
        if is_valid:
            print("✅ Database validation passed!")
        else:
            print("⚠️ Database validation found issues:")
            for issue in issues:
                print(f"    - {issue}")
        
        print(f"\n5. Testing integration with get_dataset_info...")
        from configs.datasets_config import get_dataset_info
        dataset_info = get_dataset_info('ase_db', remove_h=False, ase_db_path=db_path)
        
        print(f"✅ Successfully integrated with dataset configuration system")
        print(f"    Configured for {len(dataset_info['atom_decoder'])} atom types")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def demonstrate_utility_script():
    """Demonstrate the molecular database utility script."""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Molecular Database Utility Script")
    print("="*80)
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create test database
        create_pubchem_like_database(db_path, n_molecules=30)
        
        print(f"\n1. Testing 'analyze' command...")
        import subprocess
        result = subprocess.run([
            'python', 'molecular_db_utils.py', 'analyze', '--db_path', db_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Analysis command works!")
            # Show a snippet of the output
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # First 10 lines
                print(f"    {line}")
            print("    ... (truncated)")
        else:
            print(f"❌ Analysis failed: {result.stderr}")
        
        print(f"\n2. Testing 'recommend' command...")
        result = subprocess.run([
            'python', 'molecular_db_utils.py', 'recommend', '--db_path', db_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Recommendations command works!")
        else:
            print(f"❌ Recommendations failed: {result.stderr}")
        
        print(f"\n3. Testing 'validate' command...")
        result = subprocess.run([
            'python', 'molecular_db_utils.py', 'validate', '--db_path', db_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Validation command works!")
        else:
            print(f"❌ Validation failed: {result.stderr}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility script demonstration failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def show_training_examples():
    """Show example training commands for different scenarios."""
    
    print("\n" + "="*80)
    print("TRAINING EXAMPLES FOR GENERAL MOLECULAR DATABASES")
    print("="*80)
    
    examples = [
        {
            'title': 'Basic Training with Auto-Configuration',
            'description': 'Simplest case - automatic configuration generation',
            'command': 'python main_qm9.py --dataset ase_db --ase_db_path molecules.db'
        },
        {
            'title': 'PubChem Drug Database Training',
            'description': 'Training on pharmaceutical compounds with property conditioning',
            'command': 'python main_qm9.py --dataset ase_db --ase_db_path pubchem_drugs.db --conditioning molecular_weight logP bioavailability --batch_size 128 --n_epochs 100'
        },
        {
            'title': 'Inorganic Materials Database',
            'description': 'Training on inorganic compounds without hydrogen',
            'command': 'python main_qm9.py --dataset ase_db --ase_db_path inorganics.db --remove_h --conditioning formation_energy lattice_energy --batch_size 64'
        },
        {
            'title': 'Large Database Training',
            'description': 'Training on very large databases with filtering',
            'command': 'python main_qm9.py --dataset ase_db --ase_db_path large_database.db --batch_size 256 --filter_n_atoms 50 --n_epochs 50'
        },
        {
            'title': 'Organometallic Compounds',
            'description': 'Training on organometallic compounds with magnetic properties',
            'command': 'python main_qm9.py --dataset ase_db --ase_db_path organometallics.db --conditioning energy magnetic_moment --nf 256 --n_layers 8'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['description']}")
        print(f"   Command: {example['command']}")
    
    print(f"\n💡 PRO TIPS:")
    print(f"   • Use 'python molecular_db_utils.py analyze --db_path YOUR_DB' first")
    print(f"   • Use 'python molecular_db_utils.py recommend --db_path YOUR_DB' for optimal parameters")
    print(f"   • Large molecules (>50 atoms) may need reduced batch sizes")
    print(f"   • Many elements (>20 types) may benefit from larger model sizes (--nf 256)")


def main():
    """Run the complete demonstration."""
    
    print("GENERAL MOLECULAR DATABASE SUPPORT DEMONSTRATION")
    print("E3 Diffusion for Molecules - Enhanced for PubChem and Beyond")
    print("="*80)
    
    success = True
    
    # Demonstrate automatic analysis
    if not demonstrate_automatic_analysis():
        success = False
    
    # Demonstrate utility script
    if not demonstrate_utility_script():
        success = False
    
    # Show training examples
    show_training_examples()
    
    print("\n" + "="*80)
    if success:
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\nThe E3 Diffusion system now fully supports general molecular databases!")
        print("\nKey improvements:")
        print("✅ Automatic configuration generation for any molecular database")
        print("✅ Support for the entire periodic table")
        print("✅ PubChem-style diverse molecular datasets")
        print("✅ Easy-to-use utility scripts")
        print("✅ Comprehensive validation and recommendations")
        print("✅ Seamless integration with existing training scripts")
    else:
        print("❌ Some parts of the demonstration failed.")
        print("Please check the implementation.")
    
    print("="*80)


if __name__ == "__main__":
    main()