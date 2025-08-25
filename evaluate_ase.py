#!/usr/bin/env python3
"""
Evaluation script for molecules generated from ASE-trained models.
This script provides the same functionality as eval_analyze.py but for ASE datasets.
"""

import argparse
import torch
import os
import numpy as np
from typing import Dict, Any, List, Tuple

# Import E3 Diffusion components
from qm9.analyze import analyze_stability_for_molecules
from qm9.ase_dataset import get_ase_dataset_info
from qm9.openbabel_functions import BasicMolecularMetricsOB


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate generated molecules from ASE model')
    
    # Input parameters
    parser.add_argument('--generated_molecules', type=str, required=True,
                        help='Path to generated molecules (.pt file)')
    parser.add_argument('--reference_db', type=str, required=True,
                        help='Path to reference ASE database')
    
    # Output parameters
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt',
                        help='Output file for results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as generated molecules)')
    
    # Evaluation parameters
    parser.add_argument('--n_eval_samples', type=int, default=-1,
                        help='Number of samples to evaluate (-1 for all)')
    parser.add_argument('--compute_novelty', action='store_true',
                        help='Compute novelty (requires reference dataset fingerprints)')
    
    # Analysis parameters
    parser.add_argument('--detailed_analysis', action='store_true',
                        help='Perform detailed molecular analysis')
    parser.add_argument('--save_invalid', action='store_true',
                        help='Save invalid molecules for inspection')
    
    return parser.parse_args()


def load_generated_molecules(molecules_path: str, n_samples: int = -1) -> Dict[str, torch.Tensor]:
    """Load generated molecules from file."""
    print(f"Loading generated molecules from: {molecules_path}")
    
    if not os.path.exists(molecules_path):
        raise FileNotFoundError(f"Generated molecules file not found: {molecules_path}")
    
    generated_molecules = torch.load(molecules_path, map_location='cpu')
    
    total_molecules = generated_molecules['one_hot'].shape[0]
    print(f"Loaded {total_molecules} generated molecules")
    
    # Subsample if requested
    if n_samples > 0 and n_samples < total_molecules:
        indices = torch.randperm(total_molecules)[:n_samples]
        for key in generated_molecules:
            generated_molecules[key] = generated_molecules[key][indices]
        print(f"Using {n_samples} samples for evaluation")
    
    return generated_molecules


def convert_to_molecule_list(generated_molecules: Dict[str, torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert generated molecules to list format for analysis."""
    molecule_list = []
    n_molecules = generated_molecules['one_hot'].shape[0]
    
    for i in range(n_molecules):
        # Extract molecule data
        one_hot = generated_molecules['one_hot'][i]
        positions = generated_molecules['x'][i]
        mask = generated_molecules['node_mask'][i]
        
        # Convert to atom types
        atom_types = torch.argmax(one_hot, dim=-1)
        
        # Apply mask to get valid atoms
        n_atoms = int(mask.sum())
        if n_atoms > 0:
            valid_positions = positions[:n_atoms]
            valid_atom_types = atom_types[:n_atoms]
            molecule_list.append((valid_positions, valid_atom_types))
    
    print(f"Converted {len(molecule_list)} valid molecules for analysis")
    return molecule_list


def analyze_molecular_stability(generated_molecules: Dict[str, torch.Tensor], 
                              dataset_info: Dict[str, Any]) -> Dict[str, float]:
    """Analyze molecular stability using internal stability functions."""
    print("Analyzing molecular stability...")
    
    try:
        validity_dict, _ = analyze_stability_for_molecules(generated_molecules, dataset_info)
        
        print(f"Stability analysis completed:")
        print(f"  Molecular stability: {validity_dict.get('mol_stable', 0):.2%}")
        print(f"  Atomic stability: {validity_dict.get('atm_stable', 0):.2%}")
        
        return validity_dict
    
    except Exception as e:
        print(f"Warning: Stability analysis failed: {e}")
        return {'mol_stable': 0.0, 'atm_stable': 0.0}


def compute_molecular_metrics(molecule_list: List[Tuple[torch.Tensor, torch.Tensor]], 
                            dataset_info: Dict[str, Any],
                            compute_novelty: bool = False) -> Tuple[float, float, float]:
    """Compute validity, uniqueness, and novelty using OpenBabel."""
    print("Computing molecular metrics with OpenBabel...")
    
    try:
        # Initialize metrics calculator
        ob_metrics = BasicMolecularMetricsOB(dataset_info)
        
        # Compute metrics
        validity, uniqueness, novelty = ob_metrics.evaluate(molecule_list)
        
        print(f"Molecular metrics computed:")
        print(f"  Validity: {validity:.2%}")
        print(f"  Uniqueness: {uniqueness:.2%}")
        print(f"  Novelty: {novelty:.2%}")
        
        return validity, uniqueness, novelty
    
    except Exception as e:
        print(f"Warning: Molecular metrics computation failed: {e}")
        return 0.0, 0.0, 0.0


def analyze_molecular_properties(generated_molecules: Dict[str, torch.Tensor],
                               dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze properties of generated molecules."""
    print("Analyzing molecular properties...")
    
    atom_decoder = dataset_info['atom_decoder']
    n_molecules = generated_molecules['one_hot'].shape[0]
    
    # Initialize analysis results
    analysis = {
        'molecule_sizes': [],
        'atom_counts': {element: 0 for element in atom_decoder},
        'size_distribution': {},
        'connectivity_stats': {},
        'geometric_stats': {}
    }
    
    for i in range(n_molecules):
        mask = generated_molecules['node_mask'][i]
        one_hot = generated_molecules['one_hot'][i]
        positions = generated_molecules['x'][i]
        
        n_atoms = int(mask.sum())
        analysis['molecule_sizes'].append(n_atoms)
        
        if n_atoms > 0:
            # Count atom types
            atom_types = torch.argmax(one_hot[:n_atoms], dim=-1)
            for atom_type in atom_types:
                if atom_type < len(atom_decoder):
                    analysis['atom_counts'][atom_decoder[atom_type]] += 1
            
            # Calculate geometric properties
            valid_positions = positions[:n_atoms]
            
            # Center of mass
            com = torch.mean(valid_positions, dim=0)
            
            # Radius of gyration
            distances_from_com = torch.norm(valid_positions - com, dim=1)
            radius_of_gyration = torch.sqrt(torch.mean(distances_from_com ** 2))
            
            # Pairwise distances
            if n_atoms > 1:
                pairwise_distances = torch.cdist(valid_positions.unsqueeze(0), 
                                               valid_positions.unsqueeze(0)).squeeze(0)
                # Remove diagonal (zero distances)
                pairwise_distances = pairwise_distances[pairwise_distances > 0]
                min_distance = torch.min(pairwise_distances)
                max_distance = torch.max(pairwise_distances)
                mean_distance = torch.mean(pairwise_distances)
            else:
                min_distance = max_distance = mean_distance = 0.0
            
            # Store geometric stats (simplified aggregation)
            if 'radius_of_gyration' not in analysis['geometric_stats']:
                analysis['geometric_stats']['radius_of_gyration'] = []
                analysis['geometric_stats']['min_distance'] = []
                analysis['geometric_stats']['max_distance'] = []
                analysis['geometric_stats']['mean_distance'] = []
            
            analysis['geometric_stats']['radius_of_gyration'].append(float(radius_of_gyration))
            analysis['geometric_stats']['min_distance'].append(float(min_distance))
            analysis['geometric_stats']['max_distance'].append(float(max_distance))
            analysis['geometric_stats']['mean_distance'].append(float(mean_distance))
    
    # Compute size distribution
    from collections import Counter
    size_counts = Counter(analysis['molecule_sizes'])
    analysis['size_distribution'] = dict(size_counts)
    
    # Compute geometric statistics
    for stat_name, values in analysis['geometric_stats'].items():
        if values:
            analysis['geometric_stats'][stat_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    print(f"Molecular properties analyzed for {n_molecules} molecules")
    return analysis


def save_invalid_molecules(molecule_list: List[Tuple[torch.Tensor, torch.Tensor]],
                          dataset_info: Dict[str, Any],
                          output_dir: str,
                          validity_results: List[bool]):
    """Save invalid molecules for inspection."""
    invalid_dir = os.path.join(output_dir, 'invalid_molecules')
    os.makedirs(invalid_dir, exist_ok=True)
    
    invalid_count = 0
    for i, (is_valid, (positions, atom_types)) in enumerate(zip(validity_results, molecule_list)):
        if not is_valid:
            # Save as simple text file
            with open(os.path.join(invalid_dir, f'invalid_{invalid_count:04d}.txt'), 'w') as f:
                f.write(f"Invalid Molecule {i}\n")
                f.write(f"Atoms: {len(atom_types)}\n")
                f.write("Coordinates:\n")
                for j, (pos, atom_type) in enumerate(zip(positions, atom_types)):
                    element = dataset_info['atom_decoder'][atom_type] if atom_type < len(dataset_info['atom_decoder']) else 'X'
                    f.write(f"{element} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            invalid_count += 1
    
    print(f"Saved {invalid_count} invalid molecules to {invalid_dir}")


def generate_evaluation_report(validity_dict: Dict[str, float],
                             metrics: Tuple[float, float, float],
                             properties: Dict[str, Any],
                             dataset_info: Dict[str, Any],
                             args) -> str:
    """Generate comprehensive evaluation report."""
    validity, uniqueness, novelty = metrics
    
    # Calculate derived metrics
    n_molecules = len(properties['molecule_sizes'])
    valid_molecules = int(validity * n_molecules)
    unique_valid_molecules = int(validity * uniqueness * n_molecules)
    novel_molecules = int(validity * novelty * n_molecules)
    
    report = f"""
E3 Diffusion for Molecules - ASE Dataset Evaluation Report
==========================================================

Input Information:
- Generated molecules file: {args.generated_molecules}
- Reference database: {args.reference_db}
- Total molecules evaluated: {n_molecules}

Dataset Information:
- Atom types: {dataset_info['atom_decoder']}
- Maximum atoms: {dataset_info['max_n_nodes']}

Stability Analysis:
- Molecular stability: {validity_dict.get('mol_stable', 0):.2%}
- Atomic stability: {validity_dict.get('atm_stable', 0):.2%}

Quality Metrics (OpenBabel):
- Validity: {validity:.2%} ({valid_molecules}/{n_molecules} molecules)
- Uniqueness: {uniqueness:.2%} ({unique_valid_molecules}/{valid_molecules} unique valid molecules)
- Novelty: {novelty:.2%} ({novel_molecules}/{valid_molecules} novel valid molecules)

Molecular Size Distribution:
- Average size: {np.mean(properties['molecule_sizes']):.1f} atoms
- Size range: {min(properties['molecule_sizes'])} - {max(properties['molecule_sizes'])} atoms
- Most common sizes:"""

    # Add size distribution
    sorted_sizes = sorted(properties['size_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
    for size, count in sorted_sizes:
        percentage = count / n_molecules * 100
        report += f"\n  {size} atoms: {count} molecules ({percentage:.1f}%)"

    report += f"\n\nAtom Type Distribution:"
    total_atoms = sum(properties['atom_counts'].values())
    for element, count in properties['atom_counts'].items():
        if count > 0:
            percentage = count / total_atoms * 100
            report += f"\n  {element}: {count} atoms ({percentage:.1f}%)"

    # Add geometric statistics if available
    if properties['geometric_stats']:
        report += f"\n\nGeometric Properties:"
        for stat_name, stats in properties['geometric_stats'].items():
            if isinstance(stats, dict):
                report += f"\n  {stat_name.replace('_', ' ').title()}:"
                report += f"\n    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}"
                report += f"\n    Range: {stats['min']:.3f} - {stats['max']:.3f}"

    # Summary
    report += f"\n\nSummary:
- Generated {n_molecules} molecules
- {valid_molecules} valid molecules ({validity:.1%})
- {unique_valid_molecules} unique valid molecules
- {novel_molecules} novel molecules
- Average molecular size: {np.mean(properties['molecule_sizes']):.1f} atoms

Quality Score: {validity * uniqueness:.3f} (validity × uniqueness)
"""

    return report


def main():
    """Main function."""
    # Setup
    args = setup_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.generated_molecules)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    generated_molecules = load_generated_molecules(args.generated_molecules, args.n_eval_samples)
    dataset_info = get_ase_dataset_info(args.reference_db)
    
    # Convert to molecule list for analysis
    molecule_list = convert_to_molecule_list(generated_molecules)
    
    # Analyze molecular stability
    validity_dict = analyze_molecular_stability(generated_molecules, dataset_info)
    
    # Compute molecular metrics
    metrics = compute_molecular_metrics(molecule_list, dataset_info, args.compute_novelty)
    
    # Analyze molecular properties
    if args.detailed_analysis:
        properties = analyze_molecular_properties(generated_molecules, dataset_info)
    else:
        # Basic property analysis
        properties = {
            'molecule_sizes': [int(mask.sum()) for mask in generated_molecules['node_mask']],
            'atom_counts': {element: 0 for element in dataset_info['atom_decoder']},
            'size_distribution': {},
            'geometric_stats': {}
        }
        
        # Basic atom counting
        for i in range(generated_molecules['one_hot'].shape[0]):
            mask = generated_molecules['node_mask'][i]
            one_hot = generated_molecules['one_hot'][i]
            n_atoms = int(mask.sum())
            
            if n_atoms > 0:
                atom_types = torch.argmax(one_hot[:n_atoms], dim=-1)
                for atom_type in atom_types:
                    if atom_type < len(dataset_info['atom_decoder']):
                        properties['atom_counts'][dataset_info['atom_decoder'][atom_type]] += 1
        
        from collections import Counter
        properties['size_distribution'] = dict(Counter(properties['molecule_sizes']))
    
    # Generate report
    report = generate_evaluation_report(validity_dict, metrics, properties, dataset_info, args)
    
    # Save report
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, 'w') as f:
        f.write(report)
    
    # Print report
    print(report)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Save invalid molecules if requested
    if args.save_invalid:
        # Note: This would require modification of the metrics calculation to return validity per molecule
        print("Note: Saving invalid molecules requires additional implementation")


if __name__ == "__main__":
    main()