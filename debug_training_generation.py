#!/usr/bin/env python3
"""
Debug Script for Training and Generation Issues

This script should be run to debug the specific problems mentioned in the issue:
1. Only P atoms (19 atoms) for conditional generation
2. Only S atoms (50 atoms) for chain generation  
3. S atoms clustered around ±49 coordinates plus many Br atoms at (0,0,0) for molecule generation

Usage:
    python debug_training_generation.py

This will help identify the root causes and verify that fixes are working.
"""

import sys
import os
sys.path.append('/home/runner/work/e3_diffusion_for_molecules/e3_diffusion_for_molecules')

import torch
import numpy as np
from collections import Counter

def analyze_generated_molecules(one_hot, charges, x, node_mask, dataset_info):
    """
    Analyze the generated molecules for the specific issues mentioned in the problem statement
    """
    print("=" * 80)
    print("ANALYZING GENERATED MOLECULES")
    print("=" * 80)
    
    batch_size = one_hot.size(0)
    atom_decoder = dataset_info['atom_decoder']
    
    issues_found = []
    
    for i in range(batch_size):
        # Get the actual number of atoms for this molecule
        n_atoms = int(node_mask[i].sum().item())
        
        # Get atom types for active atoms only
        if one_hot.dim() == 3:
            # One-hot encoded
            atom_indices = torch.argmax(one_hot[i, :n_atoms], dim=1)
        else:
            # Already indices
            atom_indices = one_hot[i, :n_atoms]
        
        atom_types = [atom_decoder[idx.item()] for idx in atom_indices]
        atom_counts = Counter(atom_types)
        
        # Get coordinates for active atoms only
        coords = x[i, :n_atoms].detach().cpu().numpy()
        
        print(f"\nMolecule {i+1}:")
        print(f"  Number of atoms: {n_atoms}")
        print(f"  Atom composition: {dict(atom_counts)}")
        print(f"  Coordinate range: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
              f"Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
              f"Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
        
        # Check for the specific issues mentioned in the problem statement
        
        # Issue 1: Only P atoms (mentioned for conditional generation)
        if len(atom_counts) == 1 and 'P' in atom_counts:
            print(f"  ⚠️  ISSUE 1 DETECTED: Only P atoms generated")
            issues_found.append(f"Molecule {i+1}: Only P atoms")
        
        # Issue 2: Only S atoms (mentioned for chain generation)
        if len(atom_counts) == 1 and 'S' in atom_counts:
            print(f"  ⚠️  ISSUE 2 DETECTED: Only S atoms generated")
            issues_found.append(f"Molecule {i+1}: Only S atoms")
        
        # Issue 3: Coordinates clustered around ±49
        coord_abs_max = np.max(np.abs(coords))
        if coord_abs_max > 40:
            print(f"  ⚠️  ISSUE 3 DETECTED: Large coordinates (max: {coord_abs_max:.2f})")
            issues_found.append(f"Molecule {i+1}: Large coordinates ({coord_abs_max:.2f})")
        
        # Issue 4: Many Br atoms at (0,0,0) in unused positions
        if one_hot.size(1) > n_atoms:  # There are unused positions
            unused_positions = one_hot.size(1) - n_atoms
            
            # Check unused positions for Br atoms
            if one_hot.dim() == 3:
                unused_one_hot = one_hot[i, n_atoms:]
                br_index = atom_decoder.index('Br') if 'Br' in atom_decoder else -1
                
                if br_index >= 0:
                    unused_br_count = torch.sum(unused_one_hot[:, br_index]).item()
                    if unused_br_count > 0:
                        print(f"  ⚠️  ISSUE 4 DETECTED: {unused_br_count} Br atoms in {unused_positions} unused positions")
                        issues_found.append(f"Molecule {i+1}: Br atoms in unused positions")
            
            # Check for atoms at exactly (0,0,0) in unused positions
            unused_coords = x[i, n_atoms:].detach().cpu().numpy()
            atoms_at_origin = np.sum(np.all(np.abs(unused_coords) < 1e-6, axis=1))
            if atoms_at_origin > 0:
                print(f"  ⚠️  ISSUE 4b DETECTED: {atoms_at_origin} atoms at origin in unused positions")
                issues_found.append(f"Molecule {i+1}: Atoms at origin in unused positions")
    
    print(f"\n" + "=" * 80)
    print("ISSUE SUMMARY")
    print("=" * 80)
    
    if not issues_found:
        print("✅ NO ISSUES DETECTED - Generation appears to be working correctly!")
    else:
        print(f"❌ {len(issues_found)} ISSUES DETECTED:")
        for issue in issues_found:
            print(f"  - {issue}")
    
    return issues_found

def debug_context_preparation(conditioning, minibatch, property_norms):
    """
    Debug context preparation to check for bias issues
    """
    print("=" * 80)
    print("DEBUGGING CONTEXT PREPARATION")
    print("=" * 80)
    
    from qm9.utils import prepare_context
    
    try:
        context = prepare_context(conditioning, minibatch, property_norms)
        print(f"✅ Context preparation successful")
        print(f"   Shape: {context.shape}")
        print(f"   Range: [{context.min().item():.4f}, {context.max().item():.4f}]")
        print(f"   Mean: {context.mean().item():.4f}")
        print(f"   Std: {context.std().item():.4f}")
        
        # Check for extreme values that could cause bias
        extreme_values = torch.sum(torch.abs(context) > 5.0).item()
        if extreme_values > 0:
            print(f"  ⚠️  WARNING: {extreme_values} extreme values (>5.0) detected")
        
        return context
        
    except Exception as e:
        print(f"❌ Context preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_sampling_context(args, device, dataset_info):
    """
    Debug sampling context initialization
    """
    print("=" * 80)
    print("DEBUGGING SAMPLING CONTEXT INITIALIZATION")
    print("=" * 80)
    
    if not hasattr(args, 'context_node_nf') or args.context_node_nf == 0:
        print("No conditioning features, skipping context debug")
        return None
    
    try:
        # Simulate context initialization as done in sampling
        n_samples = 2
        n_nodes = 19 if 'qm9' in args.dataset else 50
        
        context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
        
        # Apply the same logic as in the fixed sampling functions
        feature_start_idx = 0
        
        if hasattr(args, 'conditioning'):
            for feat in args.conditioning:
                if feat == 'atom_types_encoding':
                    n_atom_types = len(dataset_info['atom_decoder'])
                    
                    # Check if we're using the realistic distribution
                    realistic_dist = torch.tensor([
                        0.45, 0.40, 0.08, 0.05, 0.01,  # H, C, N, O, F
                    ] + [0.001] * (n_atom_types - 5)).to(device)
                    
                    normalized_dist = (realistic_dist - realistic_dist.mean()) * 0.1
                    
                    for i in range(n_atom_types):
                        if feature_start_idx + i < args.context_node_nf:
                            context[:, :, feature_start_idx + i] = normalized_dist[i]
                    
                    print(f"  {feat}: Using realistic atom type distribution")
                    print(f"    Distribution: {normalized_dist[:5].tolist()} (showing first 5)")
                    
                    feature_start_idx += n_atom_types
                
                elif feat in ['molecular_weight', 'pi_conjugation_ratio']:
                    feature_start_idx += 1
                elif feat == 'functional_groups_encoding':
                    feature_start_idx += 10  # Estimated
        
        print(f"✅ Sampling context initialization successful")
        print(f"   Shape: {context.shape}")
        print(f"   Range: [{context.min().item():.4f}, {context.max().item():.4f}]")
        
        return context
        
    except Exception as e:
        print(f"❌ Sampling context initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_normalization_factors(args):
    """
    Check normalization factors for potential coordinate scaling issues
    """
    print("=" * 80)
    print("CHECKING NORMALIZATION FACTORS")
    print("=" * 80)
    
    print(f"Current normalize_factors: {args.normalize_factors}")
    
    # Check for the ±49 coordinate issue
    if args.normalize_factors[0] != 1:
        print(f"⚠️  WARNING: X normalization factor is {args.normalize_factors[0]}, should be 1")
        print(f"   This could cause coordinate scaling issues (±49 clustering)")
    else:
        print(f"✅ X normalization factor is correct (1)")
    
    # Check categorical normalization for ASE databases
    if 'ase_db' in args.dataset:
        if args.normalize_factors[1] > 2:
            print(f"⚠️  WARNING: Categorical normalization factor is {args.normalize_factors[1]}")
            print(f"   For ASE databases with many elements, this should be closer to 1")
        else:
            print(f"✅ Categorical normalization factor is reasonable ({args.normalize_factors[1]})")

def main():
    """
    Main debug function that can be called during training
    """
    print("E3 DIFFUSION DEBUG SCRIPT")
    print("This script helps debug the specific training and generation issues")
    print("mentioned in the problem statement.")
    print("")
    
    # This function can be imported and called during training
    # Example usage:
    print("USAGE EXAMPLES:")
    print("")
    print("1. Debug context preparation:")
    print("   debug_context_preparation(args.conditioning, minibatch, property_norms)")
    print("")
    print("2. Analyze generated molecules:")
    print("   issues = analyze_generated_molecules(one_hot, charges, x, node_mask, dataset_info)")
    print("")
    print("3. Debug sampling context:")
    print("   debug_sampling_context(args, device, dataset_info)")
    print("")
    print("4. Check normalization factors:")
    print("   check_normalization_factors(args)")
    print("")
    print("Call these functions at appropriate points in your training/generation code")
    print("to debug the specific issues mentioned in the problem statement.")

if __name__ == "__main__":
    main()