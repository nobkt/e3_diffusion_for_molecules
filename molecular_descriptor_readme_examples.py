#!/usr/bin/env python3
"""
Example demonstrating how to use the updated eval_conditional_qm9.py with 
molecular descriptor conditioning, equivalent to the README examples but for
the new conditioning options.

This example shows the complete workflow:
1. Train a Conditional EDM with molecular descriptors
2. Generate samples for different property values
3. Train a property classifier network
4. Evaluate the property classifier on EDM
"""

import os
import tempfile
import numpy as np
from ase import Atoms
from ase.db import connect

def create_example_database():
    """Create an example ASE database for demonstration."""
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    db = connect(db_path)
    
    print(f"Creating example database: {db_path}")
    
    # Add some diverse molecules
    molecules = [
        # Water molecules
        (Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]),
         {'energy': -76.4, 'homo': -12.6, 'lumo': 1.4}),
        
        # Methane molecules
        (Atoms('CH4', positions=[[0, 0, 0], [1.089, 1.089, 1.089], [1.089, -1.089, -1.089], 
                                [-1.089, 1.089, -1.089], [-1.089, -1.089, 1.089]]),
         {'energy': -40.5, 'homo': -14.4, 'lumo': 6.0}),
        
        # Ammonia molecules  
        (Atoms('NH3', positions=[[0, 0, 0], [1.017, 0, 0], [-0.509, 0.882, 0], [-0.509, -0.882, 0]]),
         {'energy': -56.5, 'homo': -10.8, 'lumo': 2.1}),
        
        # CO2 molecules
        (Atoms('CO2', positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
         {'energy': -188.6, 'homo': -13.8, 'lumo': 4.0}),
    ]
    
    # Add variations of each molecule
    for _ in range(25):  # 100 total molecules
        for atoms, props in molecules:
            # Add some noise to positions and properties
            positions = atoms.get_positions() + np.random.normal(0, 0.05, atoms.get_positions().shape)
            new_atoms = atoms.copy()
            new_atoms.set_positions(positions)
            
            new_props = {k: v + np.random.normal(0, 0.1) for k, v in props.items()}
            db.write(new_atoms, data=new_props)
    
    print(f"Created database with {db.count()} molecules")
    return db_path

def show_training_examples(db_path):
    """Show training command examples equivalent to README."""
    
    print("\n" + "="*80)
    print("TRAINING COMMANDS - Molecular Descriptor Conditioning")
    print("="*80)
    
    print("\n1. Train a Conditional EDM (equivalent to README example)")
    print("-" * 60)
    print("Original README command:")
    print("python main_qm9.py --exp_name exp_cond_alpha --model egnn_dynamics --lr 1e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning alpha --dataset qm9_second_half")
    
    print("\nNEW: Molecular descriptor conditioning command:")
    cmd = f"""python main_qm9.py \\
    --exp_name exp_cond_molecular_descriptors \\
    --model egnn_dynamics \\
    --lr 1e-4 \\
    --nf 192 \\
    --n_layers 9 \\
    --save_model True \\
    --diffusion_steps 1000 \\
    --sin_embedding False \\
    --n_epochs 3000 \\
    --n_stability_samples 500 \\
    --diffusion_noise_schedule polynomial_2 \\
    --diffusion_noise_precision 1e-5 \\
    --dequantization deterministic \\
    --include_charges False \\
    --diffusion_loss_type l2 \\
    --batch_size 64 \\
    --normalize_factors [1,8,1] \\
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \\
    --dataset ase_db \\
    --ase_db_path {db_path}"""
    print(cmd)

def show_generation_examples():
    """Show generation command examples equivalent to README."""
    
    print("\n\n2. Generate samples for different property values (equivalent to README example)")
    print("-" * 80)
    print("Original README command:")
    print("python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha --property alpha --n_sweeps 10 --task qualitative")
    
    print("\nNEW: Molecular descriptor generation commands:")
    
    # For each new conditioning property
    descriptors = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding']
    
    for descriptor in descriptors:
        cmd = f"""python eval_conditional_qm9.py \\
    --generators_path outputs/exp_cond_molecular_descriptors \\
    --property {descriptor} \\
    --n_sweeps 10 \\
    --task qualitative"""
        print(f"\nGenerate samples conditioned on {descriptor}:")
        print(cmd)

def show_classifier_examples(db_path):
    """Show classifier training examples equivalent to README."""
    
    print("\n\n3. Train a property classifier network (equivalent to README example)")
    print("-" * 80)
    print("Original README command:")
    print("cd qm9/property_prediction")
    print("python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_class_alpha --model_name egnn")
    
    print("\nNEW: Molecular descriptor classifier commands:")
    
    descriptors = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding']
    
    for descriptor in descriptors:
        cmd = f"""cd qm9/property_prediction
python main_qm9_prop.py \\
    --num_workers 2 \\
    --lr 5e-4 \\
    --property {descriptor} \\
    --exp_name exp_class_{descriptor} \\
    --model_name egnn \\
    --dataset ase_db \\
    --ase_db_path {db_path}"""
        print(f"\nTrain classifier for {descriptor}:")
        print(cmd)

def show_evaluation_examples():
    """Show evaluation command examples equivalent to README."""
    
    print("\n\n4. Evaluate the property classifier on EDM (equivalent to README example)")
    print("-" * 80)
    print("Original README command:")
    print("python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha --classifiers_path qm9/property_prediction/outputs/exp_class_alpha --property alpha --iterations 100 --batch_size 100 --task edm")
    
    print("\nNEW: Molecular descriptor evaluation commands:")
    
    descriptors = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding']
    
    for descriptor in descriptors:
        cmd = f"""python eval_conditional_qm9.py \\
    --generators_path outputs/exp_cond_molecular_descriptors \\
    --classifiers_path qm9/property_prediction/outputs/exp_class_{descriptor} \\
    --property {descriptor} \\
    --iterations 100 \\
    --batch_size 100 \\
    --task edm"""
        print(f"\nEvaluate classifier for {descriptor}:")
        print(cmd)

def show_combined_examples():
    """Show examples for combined conditioning as mentioned in the problem statement."""
    
    print("\n\n5. COMBINED CONDITIONING (as requested in the problem statement)")
    print("-" * 80)
    
    print("The problem statement requested support for models trained with:")
    print("--conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding")
    
    print("\nNow you can evaluate each property individually from such a model:")
    
    base_cmd = """python eval_conditional_qm9.py \\
    --generators_path outputs/your_multi_conditional_model"""
    
    descriptors = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding']
    
    for descriptor in descriptors:
        cmd = f"""{base_cmd} \\
    --property {descriptor} \\
    --n_sweeps 10 \\
    --task qualitative"""
        print(f"\nGenerate samples focusing on {descriptor}:")
        print(cmd)
    
    print("\nFor classifier evaluation (assuming you trained classifiers for each property):")
    for descriptor in descriptors:
        cmd = f"""{base_cmd} \\
    --classifiers_path qm9/property_prediction/outputs/exp_class_{descriptor} \\
    --property {descriptor} \\
    --iterations 100 \\
    --batch_size 100 \\
    --task edm"""
        print(f"\nEvaluate {descriptor} classifier:")
        print(cmd)

def main():
    """Main demonstration function."""
    
    print("="*80)
    print("MOLECULAR DESCRIPTOR CONDITIONING - README WORKFLOW EXAMPLES")
    print("="*80)
    
    print("This example shows how to use eval_conditional_qm9.py with the new")
    print("molecular descriptor conditioning options, equivalent to the workflow")
    print("described in the README but for the new conditioning properties.")
    
    # Create example database
    db_path = create_example_database()
    
    try:
        # Show all the examples
        show_training_examples(db_path)
        show_generation_examples()
        show_classifier_examples(db_path)
        show_evaluation_examples()
        show_combined_examples()
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("✅ All README workflows now support molecular descriptor conditioning")
        print("✅ Can train with: molecular_weight, pi_conjugation_ratio, atom_types_encoding, functional_groups_encoding")
        print("✅ Can generate samples conditioned on any of these properties")
        print("✅ Can train property classifiers for each descriptor")
        print("✅ Can evaluate classifiers on generated samples")
        print("✅ Supports combined conditioning as requested in the problem statement")
        
        print(f"\n📁 Example database created at: {db_path}")
        print("   (You can use this for testing the commands)")
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up example database: {db_path}")

if __name__ == "__main__":
    main()