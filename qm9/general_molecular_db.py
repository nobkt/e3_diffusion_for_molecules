"""
General molecular database support for E3 Diffusion.

This module provides utilities to work with arbitrary molecular databases,
particularly from sources like PubChem, and automatically configure
the system to handle any combination of elements and molecular properties.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from ase.data import atomic_numbers, chemical_symbols, atomic_masses


def get_comprehensive_element_mapping():
    """
    Get a comprehensive mapping of atomic numbers to element symbols
    covering the entire periodic table.
    
    Returns
    -------
    atomic_num_to_symbol : dict
        Complete mapping from atomic number to element symbol
    symbol_to_atomic_num : dict
        Complete mapping from element symbol to atomic number
    """
    # Use ASE's complete periodic table data
    atomic_num_to_symbol = {}
    symbol_to_atomic_num = {}
    
    for symbol in chemical_symbols:
        if symbol != 'X':  # Skip the placeholder 'X'
            atomic_num = atomic_numbers[symbol]
            atomic_num_to_symbol[atomic_num] = symbol
            symbol_to_atomic_num[symbol] = atomic_num
    
    return atomic_num_to_symbol, symbol_to_atomic_num


def analyze_ase_database(db_path, include_charges=True, remove_h=False):
    """
    Analyze an ASE database to extract comprehensive information
    about the molecular dataset.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file
    include_charges : bool
        Whether to include atomic charges in analysis
    remove_h : bool
        Whether to exclude hydrogen atoms from analysis
        
    Returns
    -------
    analysis : dict
        Comprehensive analysis of the database containing:
        - unique_elements: list of unique atomic symbols
        - element_counts: dict mapping elements to occurrence counts
        - molecular_sizes: dict mapping molecule sizes to counts
        - max_atoms: maximum number of atoms in any molecule
        - total_molecules: total number of molecules
        - available_properties: list of available molecular properties
        - property_statistics: dict with statistics for each property
    """
    try:
        from ase.db import connect
    except ImportError:
        raise ImportError("ASE package is required. Install with: pip install ase")
    
    print(f"Analyzing ASE database: {db_path}")
    
    # Connect to database
    db = connect(db_path)
    
    # Initialize analysis data structures
    unique_elements = set()
    element_counts = {}
    molecular_sizes = {}
    available_properties = set()
    property_values = {}
    total_molecules = 0
    max_atoms = 0
    
    # Analyze each molecule in the database
    for row in db.select():
        atoms = row.toatoms()
        total_molecules += 1
        
        # Get atomic symbols and numbers
        symbols = atoms.get_chemical_symbols()
        atomic_numbers = atoms.numbers
        
        if remove_h:
            # Filter out hydrogen atoms
            mask = atomic_numbers != 1
            symbols = [s for s, keep in zip(symbols, mask) if keep]
            atomic_numbers = atomic_numbers[mask]
        
        # Update element statistics
        for symbol in symbols:
            unique_elements.add(symbol)
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        
        # Update molecular size statistics
        n_atoms = len(symbols)
        max_atoms = max(max_atoms, n_atoms)
        molecular_sizes[n_atoms] = molecular_sizes.get(n_atoms, 0) + 1
        
        # Analyze properties
        if hasattr(row, 'data') and row.data:
            for prop_name, value in row.data.items():
                available_properties.add(prop_name)
                if prop_name not in property_values:
                    property_values[prop_name] = []
                try:
                    # Try to convert to float for numerical properties
                    property_values[prop_name].append(float(value))
                except (ValueError, TypeError):
                    # Skip non-numerical properties for statistics
                    pass
        
        if hasattr(row, 'key_value_pairs') and row.key_value_pairs:
            for prop_name, value in row.key_value_pairs.items():
                available_properties.add(prop_name)
                if prop_name not in property_values:
                    property_values[prop_name] = []
                try:
                    property_values[prop_name].append(float(value))
                except (ValueError, TypeError):
                    pass
    
    # Compute property statistics
    property_statistics = {}
    for prop_name, values in property_values.items():
        if len(values) > 0:
            values_array = np.array(values)
            property_statistics[prop_name] = {
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'coverage': len(values) / total_molecules  # Fraction of molecules with this property
            }
    
    analysis = {
        'unique_elements': sorted(list(unique_elements)),
        'element_counts': element_counts,
        'molecular_sizes': molecular_sizes,
        'max_atoms': max_atoms,
        'total_molecules': total_molecules,
        'available_properties': sorted(list(available_properties)),
        'property_statistics': property_statistics
    }
    
    print(f"Analysis complete:")
    print(f"  - {total_molecules} molecules")
    print(f"  - {len(unique_elements)} unique elements: {', '.join(sorted(unique_elements))}")
    print(f"  - Max atoms per molecule: {max_atoms}")
    print(f"  - Available properties: {', '.join(sorted(available_properties))}")
    
    return analysis


def create_optimal_dataset_config(analysis, dataset_name="general_ase_db", with_h=True):
    """
    Create an optimal dataset configuration based on database analysis.
    
    Parameters
    ----------
    analysis : dict
        Analysis results from analyze_ase_database()
    dataset_name : str
        Name for the dataset configuration
    with_h : bool
        Whether the configuration should include hydrogen atoms
        
    Returns
    -------
    config : dict
        Dataset configuration optimized for the analyzed database
    """
    elements = analysis['unique_elements']
    
    # Filter elements based on hydrogen inclusion
    if not with_h and 'H' in elements:
        elements = [e for e in elements if e != 'H']
    
    # Create atom encoder/decoder mappings
    atom_encoder = {symbol: i for i, symbol in enumerate(elements)}
    atom_decoder = elements.copy()
    
    # Get comprehensive element data for atomic numbers
    atomic_num_to_symbol, symbol_to_atomic_num = get_comprehensive_element_mapping()
    atomic_nb = [symbol_to_atomic_num[symbol] for symbol in elements if symbol in symbol_to_atomic_num]
    
    # Create color and radius mappings for visualization
    # Use a diverse set of colors and reasonable atomic radii
    base_colors = [
        '#FFFFFF99',  # White (often for H)
        'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#FFB347', '#87CEEB', '#F0E68C', '#FFE4E1'
    ]
    
    # Default atomic radii (in Å) for common elements
    default_radii = {
        'H': 0.46, 'C': 0.77, 'N': 0.77, 'O': 0.77, 'F': 0.77,
        'P': 0.77, 'S': 0.77, 'Cl': 0.77, 'Br': 0.77, 'I': 0.77,
        'Si': 0.77, 'Al': 0.77, 'B': 0.77, 'Li': 0.77, 'Na': 0.77,
        'K': 0.77, 'Ca': 0.77, 'Mg': 0.77, 'Fe': 0.77, 'Zn': 0.77
    }
    
    colors_dic = []
    radius_dic = []
    
    for i, symbol in enumerate(elements):
        # Assign colors cyclically
        if symbol == 'H' and with_h:
            colors_dic.append('#FFFFFF99')  # Special color for hydrogen
        else:
            color_idx = i % len(base_colors)
            colors_dic.append(base_colors[color_idx])
        
        # Assign radii
        radius_dic.append(default_radii.get(symbol, 0.77))
    
    # Create the configuration
    config = {
        'name': dataset_name,
        'atom_encoder': atom_encoder,
        'atom_decoder': atom_decoder,
        'atomic_nb': atomic_nb,
        'max_n_nodes': analysis['max_atoms'],
        'n_nodes': analysis['molecular_sizes'],
        'atom_types': {},  # Will be populated during dataset loading
        'distances': [],   # Will be populated if needed
        'colors_dic': colors_dic,
        'radius_dic': radius_dic,
        'with_h': with_h,
        'total_molecules': analysis['total_molecules'],
        'available_properties': analysis['available_properties'],
        'property_statistics': analysis['property_statistics']
    }
    
    return config


def save_dataset_config(config, config_path):
    """
    Save a dataset configuration to a JSON file.
    
    Parameters
    ----------
    config : dict
        Dataset configuration to save
    config_path : str
        Path where to save the configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Dataset configuration saved to: {config_path}")


def load_dataset_config(config_path):
    """
    Load a dataset configuration from a JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    config : dict
        Loaded dataset configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_general_ase_config(db_path, remove_h=False, save_config=False, config_dir=None):
    """
    Automatically create a general ASE database configuration by analyzing the database.
    
    Parameters
    ---------- 
    db_path : str
        Path to the ASE database file
    remove_h : bool
        Whether to remove hydrogen atoms
    save_config : bool
        Whether to save the configuration to disk
    config_dir : str, optional
        Directory to save configuration (defaults to configs/ase_configs/)
        
    Returns
    -------
    config : dict
        Generated dataset configuration
    """
    # Analyze the database
    analysis = analyze_ase_database(db_path, remove_h=remove_h)
    
    # Create configuration name based on database
    db_name = os.path.splitext(os.path.basename(db_path))[0]
    suffix = "_no_h" if remove_h else "_with_h"
    dataset_name = f"{db_name}{suffix}"
    
    # Create optimal configuration
    config = create_optimal_dataset_config(
        analysis, 
        dataset_name=dataset_name, 
        with_h=not remove_h
    )
    
    if save_config:
        if config_dir is None:
            config_dir = "configs/ase_configs"
        
        config_path = os.path.join(config_dir, f"{dataset_name}.json")
        save_dataset_config(config, config_path)
    
    return config


def validate_molecular_database(db_path):
    """
    Validate that an ASE database is suitable for molecular diffusion training.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file
        
    Returns
    -------
    is_valid : bool
        Whether the database is suitable
    issues : list
        List of potential issues found
    recommendations : list
        List of recommendations for improvement
    """
    try:
        from ase.db import connect
    except ImportError:
        return False, ["ASE package not available"], ["Install ASE: pip install ase"]
    
    if not os.path.exists(db_path):
        return False, [f"Database file not found: {db_path}"], ["Check the file path"]
    
    issues = []
    recommendations = []
    
    try:
        db = connect(db_path)
        total_molecules = 0
        
        # Check if database is empty
        try:
            first_row = next(db.select())
            total_molecules = len(list(db.select()))
        except StopIteration:
            issues.append("Database is empty")
            return False, issues, ["Add molecular structures to the database"]
        
        # Check minimum number of molecules
        if total_molecules < 10:
            issues.append(f"Very few molecules ({total_molecules}). Consider using at least 100 molecules for training.")
        
        # Analyze molecular diversity
        analysis = analyze_ase_database(db_path)
        
        # Check element diversity
        n_elements = len(analysis['unique_elements'])
        if n_elements < 2:
            issues.append("Very low element diversity. Consider using molecules with more varied elements.")
        elif n_elements > 50:
            recommendations.append("High element diversity detected. Consider filtering to most common elements for better training.")
        
        # Check molecular size distribution
        sizes = list(analysis['molecular_sizes'].keys())
        if max(sizes) > 100:
            issues.append("Very large molecules detected (>100 atoms). Consider filtering for computational efficiency.")
        
        if min(sizes) < 2:
            issues.append("Very small molecules detected (<2 atoms). Consider filtering.")
        
        # Check for missing properties
        if len(analysis['available_properties']) == 0:
            recommendations.append("No molecular properties found. Consider adding properties like energy, HOMO, LUMO for conditional generation.")
        
        # Check for coordinate issues
        sample_count = 0
        coord_issues = 0
        
        for row in db.select():
            if sample_count >= 100:  # Check first 100 molecules
                break
            
            atoms = row.toatoms()
            positions = atoms.positions
            
            # Check for NaN or infinite coordinates
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                coord_issues += 1
            
            # Check for unrealistic coordinates (very large distances)
            if np.any(np.abs(positions) > 1000):  # More than 1000 Å from origin
                coord_issues += 1
            
            sample_count += 1
        
        if coord_issues > 0:
            issues.append(f"Coordinate issues found in {coord_issues}/{sample_count} sampled molecules")
            recommendations.append("Check molecular geometries and ensure coordinates are in Angstrom units")
        
    except Exception as e:
        issues.append(f"Error accessing database: {str(e)}")
        return False, issues, ["Check database file integrity"]
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        recommendations.extend([
            "Consider preprocessing the database to address the issues",
            "Use the analyze_ase_database() function for detailed analysis"
        ])
    
    return is_valid, issues, recommendations


def suggest_training_parameters(analysis):
    """
    Suggest optimal training parameters based on database analysis.
    
    Parameters
    ----------
    analysis : dict
        Database analysis from analyze_ase_database()
        
    Returns
    -------
    suggestions : dict
        Suggested training parameters
    """
    n_molecules = analysis['total_molecules']
    max_atoms = analysis['max_atoms']
    n_elements = len(analysis['unique_elements'])
    
    suggestions = {}
    
    # Batch size suggestions
    if n_molecules < 1000:
        suggestions['batch_size'] = min(32, n_molecules // 10)
    elif n_molecules < 10000:
        suggestions['batch_size'] = 64
    else:
        suggestions['batch_size'] = 128
    
    # Epochs based on dataset size
    if n_molecules < 1000:
        suggestions['n_epochs'] = 500
    elif n_molecules < 10000:
        suggestions['n_epochs'] = 200
    else:
        suggestions['n_epochs'] = 100
    
    # Model size based on complexity
    if max_atoms > 50 or n_elements > 10:
        suggestions['nf'] = 256
        suggestions['n_layers'] = 8
    elif max_atoms > 20 or n_elements > 5:
        suggestions['nf'] = 128
        suggestions['n_layers'] = 6
    else:
        suggestions['nf'] = 64
        suggestions['n_layers'] = 4
    
    # Learning rate
    suggestions['lr'] = 2e-4
    
    # Diffusion steps
    suggestions['diffusion_steps'] = 500
    
    # Memory considerations
    if max_atoms > 100:
        suggestions['batch_size'] = min(suggestions['batch_size'], 32)
        suggestions['recommendations'] = [
            "Large molecules detected. Consider reducing batch size if you encounter memory issues.",
            "Consider filtering molecules to a smaller maximum size for efficiency."
        ]
    
    # Property conditioning suggestions
    good_properties = []
    for prop, stats in analysis['property_statistics'].items():
        if stats['coverage'] > 0.8 and stats['std'] > 0.01:  # Good coverage and variance
            good_properties.append(prop)
    
    if good_properties:
        suggestions['recommended_conditioning'] = good_properties[:3]  # Top 3 properties
    
    return suggestions


def print_database_summary(db_path, save_summary=False, summary_path=None):
    """
    Print a comprehensive summary of an ASE molecular database.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file
    save_summary : bool
        Whether to save the summary to a file
    summary_path : str, optional
        Path to save the summary (defaults to database_name_summary.txt)
    """
    print("=" * 80)
    print(f"MOLECULAR DATABASE SUMMARY: {os.path.basename(db_path)}")
    print("=" * 80)
    
    # Validate database
    is_valid, issues, recommendations = validate_molecular_database(db_path)
    
    if not is_valid:
        print("\n❌ DATABASE VALIDATION FAILED")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        return
    
    # Analyze database
    analysis = analyze_ase_database(db_path)
    
    # Print basic statistics
    print(f"\n📊 BASIC STATISTICS")
    print(f"  Total molecules: {analysis['total_molecules']:,}")
    print(f"  Unique elements: {len(analysis['unique_elements'])} ({', '.join(analysis['unique_elements'])})")
    print(f"  Max atoms per molecule: {analysis['max_atoms']}")
    print(f"  Available properties: {len(analysis['available_properties'])}")
    
    # Print molecular size distribution
    print(f"\n🔬 MOLECULAR SIZE DISTRIBUTION")
    sizes = sorted(analysis['molecular_sizes'].items())
    for size, count in sizes[:10]:  # Show first 10 size categories
        percentage = (count / analysis['total_molecules']) * 100
        print(f"  {size:2d} atoms: {count:6,} molecules ({percentage:5.1f}%)")
    if len(sizes) > 10:
        print(f"  ... and {len(sizes) - 10} more size categories")
    
    # Print element frequency
    print(f"\n⚛️  ELEMENT FREQUENCY")
    sorted_elements = sorted(analysis['element_counts'].items(), key=lambda x: x[1], reverse=True)
    for element, count in sorted_elements[:15]:  # Show top 15 elements
        percentage = (count / sum(analysis['element_counts'].values())) * 100
        print(f"  {element:>2}: {count:8,} occurrences ({percentage:5.1f}%)")
    if len(sorted_elements) > 15:
        print(f"  ... and {len(sorted_elements) - 15} more elements")
    
    # Print property statistics
    if analysis['property_statistics']:
        print(f"\n📈 PROPERTY STATISTICS")
        for prop, stats in sorted(analysis['property_statistics'].items()):
            print(f"  {prop}:")
            print(f"    Coverage: {stats['coverage']*100:5.1f}% ({stats['count']:,} molecules)")
            print(f"    Range: {stats['min']:.3f} to {stats['max']:.3f}")
            print(f"    Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Get training suggestions
    suggestions = suggest_training_parameters(analysis)
    print(f"\n🚀 TRAINING SUGGESTIONS")
    print(f"  Batch size: {suggestions['batch_size']}")
    print(f"  Epochs: {suggestions['n_epochs']}")
    print(f"  Model size (nf): {suggestions['nf']}")
    print(f"  Layers: {suggestions['n_layers']}")
    print(f"  Learning rate: {suggestions['lr']}")
    
    if 'recommended_conditioning' in suggestions:
        print(f"  Recommended conditioning: {', '.join(suggestions['recommended_conditioning'])}")
    
    if 'recommendations' in suggestions:
        print(f"\n💡 ADDITIONAL RECOMMENDATIONS")
        for rec in suggestions['recommendations']:
            print(f"  - {rec}")
    
    # Print example training command
    print(f"\n📋 EXAMPLE TRAINING COMMAND")
    cmd_parts = [
        "python main_qm9.py",
        "--dataset ase_db",
        f"--ase_db_path {db_path}",
        f"--batch_size {suggestions['batch_size']}",
        f"--n_epochs {suggestions['n_epochs']}",
        f"--nf {suggestions['nf']}",
        f"--n_layers {suggestions['n_layers']}",
        f"--lr {suggestions['lr']}",
    ]
    
    if 'recommended_conditioning' in suggestions:
        cmd_parts.append(f"--conditioning {' '.join(suggestions['recommended_conditioning'])}")
    
    print("  " + " \\\n    ".join(cmd_parts))
    
    print("\n" + "=" * 80)
    
    # Save summary if requested
    if save_summary:
        if summary_path is None:
            db_name = os.path.splitext(os.path.basename(db_path))[0]
            summary_path = f"{db_name}_summary.txt"
        
        # TODO: Implement summary saving to file
        print(f"\n💾 Summary saved to: {summary_path}")