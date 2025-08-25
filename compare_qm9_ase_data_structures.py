#!/usr/bin/env python3
"""
Detailed comparison script to check data structures between QM9 and ASE datasets.
This script loads actual data from both QM9 and ASE datasets and compares:
1. Tensor shapes and dimensions
2. Value ranges and statistics  
3. Data type consistency
4. Masking and edge structure
5. Normalization factors

This helps identify the exact root cause of high loss values in ASE datasets.
"""

import torch
import numpy as np
import ase.db
import ase
import tempfile
import os
import sys
from qm9 import dataset
from configs.datasets_config import get_dataset_info
import warnings

def create_test_ase_database():
    """Create a test ASE database with molecules similar to QM9."""
    db_path = '/tmp/compare_ase.db'
    
    with ase.db.connect(db_path) as db:
        # Create several test molecules with varying sizes
        test_molecules = [
            # Small molecule (3 atoms)
            {
                'symbols': ['O', 'H', 'H'],
                'positions': [[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]],
                'properties': {
                    'index': 1,
                    'U0_Ha': -76.404702,
                    'HOMO_Ha': -0.5875,
                    'LUMO_Ha': 0.0829,
                    'gap_Ha': 0.6704,
                    'alpha': 9.46,
                    'ZPVE_Ha': 0.021375,
                }
            },
            # Medium molecule (5 atoms)
            {
                'symbols': ['C', 'H', 'H', 'H', 'H'],
                'positions': [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]],
                'properties': {
                    'index': 2,
                    'U0_Ha': -40.47893,
                    'HOMO_Ha': -0.5998,
                    'LUMO_Ha': 0.0618,
                    'gap_Ha': 0.6616,
                    'alpha': 17.3,
                    'ZPVE_Ha': 0.044749,
                }
            },
            # Another medium molecule (4 atoms)
            {
                'symbols': ['N', 'H', 'H', 'H'],
                'positions': [[0, 0, 0], [0.9, 0.3, 0.3], [-0.3, 0.9, 0.3], [-0.3, -0.3, 0.9]],
                'properties': {
                    'index': 3,
                    'U0_Ha': -56.525887,
                    'HOMO_Ha': -0.4887,
                    'LUMO_Ha': 0.0278,
                    'gap_Ha': 0.5165,
                    'alpha': 14.8,
                    'ZPVE_Ha': 0.034358,
                }
            },
        ]
        
        for mol_data in test_molecules:
            atoms = ase.Atoms(mol_data['symbols'], positions=mol_data['positions'])
            db.write(atoms, **mol_data['properties'])
    
    print(f"Created test ASE database: {db_path}")
    return db_path

def load_qm9_from_local_db():
    """Load QM9-like data from local database file."""
    print("Loading QM9 dataset from local database...")
    
    # Try to use the local test_qm9.db file
    db_path = 'test_qm9.db'
    if not os.path.exists(db_path):
        db_path = 'qm9.db'
    
    if not os.path.exists(db_path):
        print(f"✗ No local QM9 database found")
        return None, None, None
    
    try:
        # Create a QM9-like dataset from the local database
        db = ase.db.connect(db_path)
        
        # Extract a few molecules to create a batch
        molecules = []
        count = 0
        for row in db.select():
            if count >= 3:  # Just get a few molecules
                break
            
            atoms = row.toatoms()
            atomic_numbers = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
            
            # Create data in QM9-like format
            mol_data = {
                'positions': torch.from_numpy(positions).float(),
                'charges': torch.from_numpy(atomic_numbers.astype(float)[:, None]).float(),
                'atom_mask': torch.ones(len(atomic_numbers), dtype=torch.bool),
            }
            
            # Create one-hot encoding (assume QM9 atom types: H=0, C=1, N=2, O=3, F=4)
            atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            n_atom_types = 5
            one_hot = torch.zeros(len(atomic_numbers), n_atom_types)
            
            # Map atomic numbers to QM9 atom types
            atomic_symbols = atoms.get_chemical_symbols()
            for i, symbol in enumerate(atomic_symbols):
                if symbol in atom_encoder:
                    one_hot[i, atom_encoder[symbol]] = 1.0
                else:
                    # Unknown atom type, use carbon as default
                    one_hot[i, 1] = 1.0
            
            mol_data['one_hot'] = one_hot
            
            # Add some mock properties
            mol_data.update({
                'U0': torch.tensor(-500.0 - count * 100),  # Mock energy in eV
                'HOMO': torch.tensor(-15.0 - count),  # Mock HOMO in eV  
                'LUMO': torch.tensor(3.0 + count),   # Mock LUMO in eV
                'gap': torch.tensor(18.0 + count),   # Mock gap in eV
                'alpha': torch.tensor(10.0 + count * 2),  # Mock polarizability
            })
            
            molecules.append(mol_data)
            count += 1
        
        if not molecules:
            print(f"✗ No molecules found in {db_path}")
            return None, None, None
        
        # Create a batch-like structure
        batch = {}
        
        # Pad molecules to same size
        max_atoms = max(mol['positions'].shape[0] for mol in molecules)
        batch_size = len(molecules)
        
        # Initialize tensors
        batch['positions'] = torch.zeros(batch_size, max_atoms, 3)
        batch['charges'] = torch.zeros(batch_size, max_atoms, 1)
        batch['one_hot'] = torch.zeros(batch_size, max_atoms, 5)
        batch['atom_mask'] = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
        
        # Fill tensors
        for i, mol in enumerate(molecules):
            n_atoms = mol['positions'].shape[0]
            batch['positions'][i, :n_atoms] = mol['positions']
            batch['charges'][i, :n_atoms] = mol['charges']
            batch['one_hot'][i, :n_atoms] = mol['one_hot']
            batch['atom_mask'][i, :n_atoms] = mol['atom_mask']
        
        # Create edge mask
        edge_mask = batch['atom_mask'].unsqueeze(1) * batch['atom_mask'].unsqueeze(2)
        diag_mask = ~torch.eye(max_atoms, dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        batch['edge_mask'] = edge_mask.view(batch_size * max_atoms * max_atoms, 1)
        
        # Add properties
        for prop in ['U0', 'HOMO', 'LUMO', 'gap', 'alpha']:
            batch[prop] = torch.stack([mol[prop] for mol in molecules])
        
        charge_scale = 4.0  # QM9 typical charge scale
        
        print(f"✓ QM9-like batch created from local database")
        print(f"  Batch size: {batch_size}")
        print(f"  Max nodes: {max_atoms}")
        print(f"  Charge scale: {charge_scale}")
        print(f"  Database: {db_path}")
        
        return batch, charge_scale, 'qm9'
        
    except Exception as e:
        print(f"✗ Failed to load from local QM9 database: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def load_qm9_batch():
    """Load a sample batch from QM9 dataset."""
    # First try normal loading, then fallback to local database
    print("Loading QM9 dataset...")
    
    class QM9Args:
        def __init__(self):
            self.dataset = 'qm9'
            self.datadir = 'qm9/temp'
            self.filter_n_atoms = None
            self.remove_h = False
            self.include_charges = True
            self.batch_size = 4
            self.num_workers = 0
            self.filter_molecule_size = None
            self.sequential = False
    
    try:
        args = QM9Args()
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        # Get first batch from training set
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"✓ QM9 batch loaded successfully")
        print(f"  Batch size: {batch['positions'].shape[0]}")
        print(f"  Max nodes: {batch['positions'].shape[1]}")
        print(f"  Charge scale: {charge_scale}")
        
        return batch, charge_scale, 'qm9'
        
    except Exception as e:
        print(f"✗ Failed to load QM9 dataset: {e}")
        print("Trying to load from local database...")
        return load_qm9_from_local_db()

def load_ase_batch():
    """Load a sample batch from ASE dataset."""
    print("\nLoading ASE dataset...")
    
    db_path = create_test_ase_database()
    
    class ASEArgs:
        def __init__(self):
            self.dataset = 'ase'
            self.ase_db_file = db_path
            self.ase_max_entries = None
            self.filter_molecule_size = None
            self.remove_h = False
            self.include_charges = True
            self.sequential = False
            self.batch_size = 3  # Use all molecules in test
            self.device = torch.device('cpu')
    
    try:
        args = ASEArgs()
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        # Get first batch from training set
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"✓ ASE batch loaded successfully")
        print(f"  Batch size: {batch['positions'].shape[0]}")
        print(f"  Max nodes: {batch['positions'].shape[1]}")
        print(f"  Charge scale: {charge_scale}")
        
        return batch, charge_scale, 'ase'
        
    except Exception as e:
        print(f"✗ Failed to load ASE dataset: {e}")
        return None, None, None
    
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def compare_tensor_structures(qm9_batch, ase_batch):
    """Compare the basic tensor structures between QM9 and ASE batches."""
    print("\n" + "="*60)
    print("TENSOR STRUCTURE COMPARISON")
    print("="*60)
    
    # Get common keys
    qm9_keys = set(qm9_batch.keys())
    ase_keys = set(ase_batch.keys())
    common_keys = qm9_keys & ase_keys
    qm9_only = qm9_keys - ase_keys
    ase_only = ase_keys - qm9_keys
    
    print(f"Common keys: {sorted(common_keys)}")
    if qm9_only:
        print(f"QM9 only keys: {sorted(qm9_only)}")
    if ase_only:
        print(f"ASE only keys: {sorted(ase_only)}")
    
    # Compare shapes and types for common keys
    print("\nTensor shape and type comparison:")
    issues_found = []
    
    for key in sorted(common_keys):
        qm9_tensor = qm9_batch[key]
        ase_tensor = ase_batch[key]
        
        # Basic tensor properties
        qm9_shape = qm9_tensor.shape if torch.is_tensor(qm9_tensor) else "not tensor"
        ase_shape = ase_tensor.shape if torch.is_tensor(ase_tensor) else "not tensor"
        
        qm9_dtype = qm9_tensor.dtype if torch.is_tensor(qm9_tensor) else type(qm9_tensor)
        ase_dtype = ase_tensor.dtype if torch.is_tensor(ase_tensor) else type(ase_tensor)
        
        print(f"  {key}:")
        print(f"    QM9: shape={qm9_shape}, dtype={qm9_dtype}")
        print(f"    ASE: shape={ase_shape}, dtype={ase_dtype}")
        
        # Check for mismatches
        if qm9_shape != ase_shape:
            issues_found.append(f"Shape mismatch for {key}: QM9={qm9_shape} vs ASE={ase_shape}")
        
        if qm9_dtype != ase_dtype:
            issues_found.append(f"Type mismatch for {key}: QM9={qm9_dtype} vs ASE={ase_dtype}")
    
    if issues_found:
        print("\n⚠️  STRUCTURAL ISSUES FOUND:")
        for issue in issues_found:
            print(f"    - {issue}")
        return False
    else:
        print("\n✓ All tensor structures match!")
        return True

def compare_value_ranges(qm9_batch, ase_batch):
    """Compare value ranges and statistics of tensors."""
    print("\n" + "="*60)
    print("VALUE RANGE COMPARISON")
    print("="*60)
    
    # Focus on key tensors that affect training
    key_tensors = ['positions', 'charges', 'one_hot', 'atom_mask', 'edge_mask']
    key_properties = ['U0_Ha', 'HOMO_Ha', 'LUMO_Ha', 'gap_Ha', 'alpha']
    
    issues_found = []
    
    print("Geometric tensors:")
    for key in key_tensors:
        if key in qm9_batch and key in ase_batch:
            qm9_tensor = qm9_batch[key]
            ase_tensor = ase_batch[key]
            
            if torch.is_tensor(qm9_tensor) and torch.is_tensor(ase_tensor):
                # Skip boolean tensors for statistics
                if qm9_tensor.dtype == torch.bool or ase_tensor.dtype == torch.bool:
                    continue
                    
                qm9_stats = {
                    'min': qm9_tensor.min().item(),
                    'max': qm9_tensor.max().item(),
                    'mean': qm9_tensor.mean().item(),
                    'std': qm9_tensor.std().item()
                }
                # Skip boolean tensors for statistics
                if ase_tensor.dtype == torch.bool:
                    continue
                    
                ase_stats = {
                    'min': ase_tensor.min().item(),
                    'max': ase_tensor.max().item(),
                    'mean': ase_tensor.mean().item(),
                    'std': ase_tensor.std().item()
                }
                
                print(f"  {key}:")
                print(f"    QM9: min={qm9_stats['min']:.4f}, max={qm9_stats['max']:.4f}, mean={qm9_stats['mean']:.4f}, std={qm9_stats['std']:.4f}")
                print(f"    ASE: min={ase_stats['min']:.4f}, max={ase_stats['max']:.4f}, mean={ase_stats['mean']:.4f}, std={ase_stats['std']:.4f}")
                
                # Check for extreme differences
                if key == 'positions':
                    if abs(qm9_stats['mean'] - ase_stats['mean']) > 1.0:
                        issues_found.append(f"Large position mean difference: QM9={qm9_stats['mean']:.4f} vs ASE={ase_stats['mean']:.4f}")
                    if qm9_stats['std'] / ase_stats['std'] > 5 or ase_stats['std'] / qm9_stats['std'] > 5:
                        issues_found.append(f"Large position std difference: QM9={qm9_stats['std']:.4f} vs ASE={ase_stats['std']:.4f}")
                
                elif key == 'charges':
                    if qm9_stats['max'] != ase_stats['max'] or qm9_stats['min'] != ase_stats['min']:
                        issues_found.append(f"Charge range mismatch: QM9=[{qm9_stats['min']:.1f}, {qm9_stats['max']:.1f}] vs ASE=[{ase_stats['min']:.1f}, {ase_stats['max']:.1f}]")
    
    print("\nMolecular properties:")
    for key in key_properties:
        if key in qm9_batch and key in ase_batch:
            qm9_tensor = qm9_batch[key]
            ase_tensor = ase_batch[key]
            
            if torch.is_tensor(qm9_tensor) and torch.is_tensor(ase_tensor):
                qm9_stats = {
                    'min': qm9_tensor.min().item(),
                    'max': qm9_tensor.max().item(),
                    'mean': qm9_tensor.mean().item(),
                }
                ase_stats = {
                    'min': ase_tensor.min().item(),
                    'max': ase_tensor.max().item(),
                    'mean': ase_tensor.mean().item(),
                }
                
                print(f"  {key}:")
                print(f"    QM9: min={qm9_stats['min']:.2f}, max={qm9_stats['max']:.2f}, mean={qm9_stats['mean']:.2f}")
                print(f"    ASE: min={ase_stats['min']:.2f}, max={ase_stats['max']:.2f}, mean={ase_stats['mean']:.2f}")
                
                # Check if properties are in expected eV ranges
                if key in ['U0_Ha', 'HOMO_Ha', 'LUMO_Ha']:
                    if abs(ase_stats['mean']) < 10:  # Properties should be in eV, not Hartree
                        issues_found.append(f"{key} appears not converted to eV: ASE mean={ase_stats['mean']:.2f}")
    
    if issues_found:
        print("\n⚠️  VALUE RANGE ISSUES FOUND:")
        for issue in issues_found:
            print(f"    - {issue}")
        return False
    else:
        print("\n✓ All value ranges are reasonable!")
        return True

def compare_masking_and_edges(qm9_batch, ase_batch):
    """Compare atom masks and edge structures."""
    print("\n" + "="*60)
    print("MASKING AND EDGE STRUCTURE COMPARISON")
    print("="*60)
    
    issues_found = []
    
    # Check atom masks
    if 'atom_mask' in qm9_batch and 'atom_mask' in ase_batch:
        qm9_mask = qm9_batch['atom_mask']
        ase_mask = ase_batch['atom_mask']
        
        qm9_num_atoms = qm9_mask.sum(dim=1)
        ase_num_atoms = ase_mask.sum(dim=1)
        
        print(f"Atom counts per molecule:")
        print(f"  QM9: {qm9_num_atoms.tolist()}")
        print(f"  ASE: {ase_num_atoms.tolist()}")
        
        # Check mask consistency with charges
        if 'charges' in qm9_batch and 'charges' in ase_batch:
            qm9_charges = qm9_batch['charges']
            ase_charges = ase_batch['charges']
            
            # QM9 mask should be charges > 0
            qm9_charge_mask = (qm9_charges.squeeze(-1) > 0)
            ase_charge_mask = (ase_charges.squeeze(-1) > 0)
            
            if not torch.allclose(qm9_mask.float(), qm9_charge_mask.float()):
                issues_found.append("QM9 atom_mask inconsistent with charges > 0")
            
            if not torch.allclose(ase_mask.float(), ase_charge_mask.float()):
                issues_found.append("ASE atom_mask inconsistent with charges > 0")
    
    # Check edge masks
    if 'edge_mask' in qm9_batch and 'edge_mask' in ase_batch:
        qm9_edges = qm9_batch['edge_mask']
        ase_edges = ase_batch['edge_mask']
        
        # Edge masks should have same total number of edges
        qm9_edge_count = qm9_edges.sum().item()
        ase_edge_count = ase_edges.sum().item()
        
        print(f"Total edge counts:")
        print(f"  QM9: {qm9_edge_count}")
        print(f"  ASE: {ase_edge_count}")
        
        # Expected edge count for each molecule should be n_atoms * (n_atoms - 1)
        if 'atom_mask' in qm9_batch:
            qm9_atoms = qm9_batch['atom_mask'].sum(dim=1)
            expected_qm9_edges = (qm9_atoms * (qm9_atoms - 1)).sum().item()
            
            if qm9_edge_count != expected_qm9_edges:
                issues_found.append(f"QM9 edge count mismatch: got {qm9_edge_count}, expected {expected_qm9_edges}")
        
        if 'atom_mask' in ase_batch:
            ase_atoms = ase_batch['atom_mask'].sum(dim=1)
            expected_ase_edges = (ase_atoms * (ase_atoms - 1)).sum().item()
            
            if ase_edge_count != expected_ase_edges:
                issues_found.append(f"ASE edge count mismatch: got {ase_edge_count}, expected {expected_ase_edges}")
    
    if issues_found:
        print("\n⚠️  MASKING/EDGE ISSUES FOUND:")
        for issue in issues_found:
            print(f"    - {issue}")
        return False
    else:
        print("\n✓ All masking and edge structures are correct!")
        return True

def compare_normalization_context(qm9_batch, ase_batch):
    """Compare potential normalization and context that affects training."""
    print("\n" + "="*60)
    print("NORMALIZATION AND TRAINING CONTEXT")
    print("="*60)
    
    # Check position centering
    if 'positions' in qm9_batch and 'positions' in ase_batch and 'atom_mask' in qm9_batch and 'atom_mask' in ase_batch:
        qm9_pos = qm9_batch['positions']
        ase_pos = ase_batch['positions']
        qm9_mask = qm9_batch['atom_mask']
        ase_mask = ase_batch['atom_mask']
        
        # Check if positions are mean-centered for each molecule
        issues_found = []
        
        for i in range(qm9_pos.shape[0]):
            mask = qm9_mask[i].unsqueeze(-1)
            masked_pos = qm9_pos[i] * mask
            n_atoms = qm9_mask[i].sum()
            if n_atoms > 1:
                mean_pos = masked_pos.sum(dim=0) / n_atoms
                if mean_pos.abs().max() > 1e-3:
                    issues_found.append(f"QM9 molecule {i} not mean-centered: mean={mean_pos}")
        
        for i in range(ase_pos.shape[0]):
            mask = ase_mask[i].unsqueeze(-1)
            masked_pos = ase_pos[i] * mask
            n_atoms = ase_mask[i].sum()
            if n_atoms > 1:
                mean_pos = masked_pos.sum(dim=0) / n_atoms
                if mean_pos.abs().max() > 1e-3:
                    issues_found.append(f"ASE molecule {i} not mean-centered: mean={mean_pos}")
        
        print("Position centering check:")
        if issues_found:
            print("  ⚠️  Issues found:")
            for issue in issues_found[:3]:  # Show first 3 issues
                print(f"    - {issue}")
            if len(issues_found) > 3:
                print(f"    - ... and {len(issues_found) - 3} more")
        else:
            print("  ✓ All molecules appear mean-centered")
    
    return len(issues_found) == 0 if 'issues_found' in locals() else True

def main():
    print("=" * 80)
    print("DETAILED QM9 vs ASE DATA STRUCTURE COMPARISON")
    print("=" * 80)
    print("This script compares the actual data structures between QM9 and ASE datasets")
    print("to identify the root cause of high loss values in ASE training.")
    print()
    
    # Load both datasets
    qm9_batch, qm9_charge_scale, qm9_name = load_qm9_batch()
    ase_batch, ase_charge_scale, ase_name = load_ase_batch()
    
    if qm9_batch is None or ase_batch is None:
        print("❌ Failed to load one or both datasets. Cannot proceed with comparison.")
        return False
    
    # Run comparisons
    struct_ok = compare_tensor_structures(qm9_batch, ase_batch)
    values_ok = compare_value_ranges(qm9_batch, ase_batch)
    masks_ok = compare_masking_and_edges(qm9_batch, ase_batch)
    norm_ok = compare_normalization_context(qm9_batch, ase_batch)
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"Tensor structures:     {'✓ PASS' if struct_ok else '❌ FAIL'}")
    print(f"Value ranges:          {'✓ PASS' if values_ok else '❌ FAIL'}")
    print(f"Masking/edges:         {'✓ PASS' if masks_ok else '❌ FAIL'}")
    print(f"Normalization:         {'✓ PASS' if norm_ok else '❌ FAIL'}")
    print(f"Charge scales:         QM9={qm9_charge_scale}, ASE={ase_charge_scale}")
    
    all_ok = struct_ok and values_ok and masks_ok and norm_ok
    
    if all_ok:
        print("\n🎉 ALL CHECKS PASSED!")
        print("The data structures appear compatible. The high loss issue may be due to:")
        print("  - Different normalization factors in training")
        print("  - Different dataset size/distribution") 
        print("  - Different molecular property ranges")
        print("  - Training hyperparameter mismatch")
    else:
        print("\n⚠️  ISSUES FOUND!")
        print("The above issues may be causing the high loss values in ASE training.")
        print("Review the specific problems and fix the data loading/processing pipeline.")
    
    print("="*80)
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)