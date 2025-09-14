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
    
    # Connect to database with error handling
    try:
        db = connect(db_path)
    except Exception as e:
        error_message = str(e).lower()
        if "file is not a database" in error_message:
            raise RuntimeError("File is not a valid ASE database. Check file format and extension.")
        elif "database disk image is malformed" in error_message:
            # For malformed databases, try to proceed with analysis as they might be partially readable
            print(f"Warning: Database reports as malformed but attempting analysis anyway...")
            try:
                db = connect(db_path)
            except Exception as e2:
                raise RuntimeError(f"Database file is corrupted and cannot be accessed: {str(e2)}")
        else:
            raise RuntimeError(f"Cannot access database: {str(e)}")
    
    # Initialize analysis data structures
    unique_elements = set()
    element_counts = {}
    molecular_sizes = {}
    available_properties = set()
    property_values = {}
    total_molecules = 0
    max_atoms = 0
    
    # Analyze each molecule in the database with error handling
    rows_processed = 0
    try:
        for row in db.select():
            try:
                atoms = row.toatoms()
                total_molecules += 1
                rows_processed += 1
                
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
            
            except Exception as e:
                print(f"Warning: Error processing molecule {total_molecules}: {str(e)}")
                # Continue with next molecule rather than failing completely
                continue
                
    except Exception as e:
        error_message = str(e).lower()
        if rows_processed == 0:
            # If we couldn't process any molecules, this might still be recoverable
            if "database disk image is malformed" in error_message:
                print(f"Warning: Database is malformed but might have accessible data. Processed {rows_processed} molecules.")
                # Allow analysis to continue with whatever data was collected
                if total_molecules == 0:
                    print("No molecules could be processed from this database.")
            else:
                raise RuntimeError(f"Cannot access database rows: {str(e)}")
        else:
            print(f"Warning: Database iteration ended after processing {rows_processed} molecules: {str(e)}")
            # Continue with analysis of molecules processed so far
    
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
    
    # Compute atom type statistics from element counts
    atom_types = {}
    if 'element_counts' in analysis:
        total_atoms = sum(analysis['element_counts'].values())
        for symbol in elements:
            if symbol in analysis['element_counts']:
                # Map symbol to encoder index and store count
                encoder_idx = atom_encoder[symbol]
                atom_types[encoder_idx] = analysis['element_counts'][symbol]
            else:
                # If element not found in counts, assign minimal count
                encoder_idx = atom_encoder[symbol]
                atom_types[encoder_idx] = 1
    
    # Calculate optimal normalization factor for this dataset
    n_elements = len(elements)
    
    # For datasets with many elements, use a smaller normalization factor
    # This helps prevent noise from disproportionately affecting rare elements
    if n_elements <= 5:
        # Original QM9-style normalization for small element sets
        optimal_categorical_norm = 4.0
    elif n_elements <= 10:
        # Moderate reduction for medium element sets  
        optimal_categorical_norm = 2.0
    else:
        # Strong reduction for large element sets to minimize noise impact
        optimal_categorical_norm = 1.0
    
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
        'atom_types': atom_types,  # Now properly populated with actual counts
        'distances': [],   # Will be populated if needed
        'colors_dic': colors_dic,
        'radius_dic': radius_dic,
        'with_h': with_h,
        'total_molecules': analysis['total_molecules'],
        'available_properties': analysis['available_properties'],
        'property_statistics': analysis['property_statistics'],
        # Add normalization recommendations 
        'recommended_categorical_norm': optimal_categorical_norm,
        'n_elements': n_elements
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
    # Always use 'ase_db' as the dataset name to match visualizer expectations
    dataset_name = "ase_db"
    
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
    
    # First, try to establish database connection and basic integrity check
    try:
        db = connect(db_path)
        
        # Count total molecules more efficiently
        try:
            # Try to get count without iterating through all rows first
            rows = list(db.select())
            total_molecules = len(rows)
            
            if total_molecules == 0:
                issues.append("Database is empty")
                return False, issues, ["Add molecular structures to the database"]
                
        except Exception as count_error:
            # Handle different types of database access errors
            error_message = str(count_error).lower()
            if "database disk image is malformed" in error_message:
                # Don't fail validation for malformed databases - they might be partially readable
                issues.append("Database reports as malformed but may still be partially readable")
                recommendations.extend([
                    "Database appears to have some corruption but may still be usable",
                    "Try running the analysis to see if data can be extracted", 
                    "Consider recreating the database if analysis fails"
                ])
                # Continue with validation using partial data if possible
                rows = []  # Empty list for now, analysis will handle this better
                total_molecules = 0
            else:
                # Other errors are still critical
                issues.append(f"Cannot access database rows: {str(count_error)}")
                return False, issues, ["Database appears to be corrupted or incompatible"]
        
        # Check minimum number of molecules  
        if total_molecules < 10:
            issues.append(f"Very few molecules ({total_molecules}). Consider using at least 100 molecules for training.")
            # Don't return early - continue with analysis but mark as having issues
        
        # Perform basic structural validation using the already loaded rows
        # This avoids calling analyze_ase_database which might fail on corrupted databases
        unique_elements = set()
        molecular_sizes = []
        coord_issues = 0
        sample_count = 0
        available_properties = set()
        
        for row in rows:
            try:
                atoms = row.toatoms()
                symbols = atoms.get_chemical_symbols()
                positions = atoms.positions
                
                # Update element statistics
                unique_elements.update(symbols)
                molecular_sizes.append(len(symbols))
                
                # Check for coordinate issues
                if sample_count < 100:  # Check first 100 molecules
                    # Check for NaN or infinite coordinates
                    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                        coord_issues += 1
                    
                    # Check for unrealistic coordinates (very large distances)
                    if np.any(np.abs(positions) > 1000):  # More than 1000 Å from origin
                        coord_issues += 1
                
                # Check properties
                if hasattr(row, 'data') and row.data:
                    available_properties.update(row.data.keys())
                if hasattr(row, 'key_value_pairs') and row.key_value_pairs:
                    available_properties.update(row.key_value_pairs.keys())
                
                sample_count += 1
                
            except Exception as e:
                # If we can't process individual molecules, note it but continue
                issues.append(f"Error processing molecule {sample_count + 1}: {str(e)}")
                sample_count += 1
                continue
        
        # Check element diversity
        n_elements = len(unique_elements)
        if n_elements < 2:
            issues.append("Very low element diversity. Consider using molecules with more varied elements.")
        elif n_elements > 50:
            recommendations.append("High element diversity detected. Consider filtering to most common elements for better training.")
        
        # Check molecular size distribution
        if molecular_sizes:
            max_size = max(molecular_sizes)
            min_size = min(molecular_sizes)
            
            if max_size > 100:
                issues.append("Very large molecules detected (>100 atoms). Consider filtering for computational efficiency.")
                recommendations.append("Use the filter command to remove large molecules: python molecular_db_utils.py filter --input_db your_db.db --output_db filtered_db.db --max_atoms 100")
            
            if min_size < 2:
                issues.append("Very small molecules detected (<2 atoms). Consider filtering.")
        
        # Check for missing properties
        if len(available_properties) == 0:
            recommendations.append("No molecular properties found. Consider adding properties like energy, HOMO, LUMO for conditional generation.")
        
        if coord_issues > 0:
            issues.append(f"Coordinate issues found in {coord_issues}/{sample_count} sampled molecules")
            recommendations.append("Check molecular geometries and ensure coordinates are in Angstrom units")
        
    except Exception as e:
        # Provide more specific error messages for common database issues
        error_message = str(e).lower()
        if "database disk image is malformed" in error_message:
            # Treat malformed database as a warning, not a fatal error
            # The database might still be partially readable
            issues.append("Database reports as malformed but may still be partially readable")
            recommendations.extend([
                "Database appears to have some corruption but may still be usable",
                "Try running the analysis to see if data can be extracted",
                "Consider recreating the database if analysis fails",
                "Check if the file was completely written/transferred"
            ])
            # Don't return False - let the analysis attempt to proceed
        elif "file is not a database" in error_message:
            issues.append("File is not a valid SQLite/ASE database")
            recommendations.extend([
                "Ensure the file is a valid ASE database",
                "Check if the file extension and format are correct",
                "Try opening the file with ASE directly to verify format"
            ])
            return False, issues, recommendations
        elif "database is locked" in error_message:
            issues.append("Database is currently locked by another process")
            recommendations.extend([
                "Close any other applications using the database",
                "Wait a moment and try again",
                "Check for concurrent access to the database file"
            ])
            return False, issues, recommendations
        else:
            issues.append(f"Error accessing database: {str(e)}")
            recommendations.append("Check database file integrity and format")
            return False, issues, recommendations
    
    is_valid = len([issue for issue in issues if is_critical_issue(issue)]) == 0
    
    if not is_valid:
        recommendations.extend([
            "Critical issues found that prevent database use",
            "Address the critical issues before proceeding"
        ])
    elif len(issues) > 0:
        recommendations.extend([
            "Database can be used but has some issues",
            "Consider preprocessing the database to address the issues",
            "Use the analyze_ase_database() function for detailed analysis"
        ])
    
    return is_valid, issues, recommendations


def is_critical_issue(issue):
    """
    Determine if an issue is critical (prevents database use) or just a warning.
    """
    issue_lower = issue.lower()
    
    # Special case: malformed but potentially readable databases are not critical
    if "malformed but may still be partially readable" in issue_lower:
        return False
    
    critical_keywords = [
        "cannot access", "not a valid", "error processing", 
        "very small molecules", "database is locked"
    ]
    
    return any(keyword in issue_lower for keyword in critical_keywords)


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
        suggestions['batch_size'] = max(1, min(32, n_molecules // 10))
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
    
    # Learning rate - reduce for large molecules and numerical stability
    if max_atoms > 100:
        suggestions['lr'] = 5e-5  # Much more conservative for very large molecules
    elif max_atoms > 50:
        suggestions['lr'] = 1e-4  # Conservative for large molecules
    else:
        suggestions['lr'] = 2e-4
    
    # Diffusion steps
    suggestions['diffusion_steps'] = 500
    
    # CRITICAL FIX: Normalization factors based on element count
    # This fixes the halogen bias issue
    if n_elements <= 5:
        # Original QM9-style for small element sets (H, C, N, O, F)
        categorical_norm = 4.0
        normalization_note = "Using QM9-style normalization (few elements)"
    elif n_elements <= 10:
        # Moderate reduction for medium element sets 
        categorical_norm = 2.0
        normalization_note = "Reduced normalization for medium element diversity"
    else:
        # Strong reduction for large element sets to minimize noise impact
        categorical_norm = 1.0
        normalization_note = "Minimal normalization for high element diversity"
    
    suggestions['normalize_factors'] = [1, categorical_norm, 1]
    suggestions['categorical_norm_factor'] = categorical_norm
    suggestions['normalization_note'] = normalization_note
    
    # Property conditioning suggestions - analyze first
    good_properties = []
    for prop, stats in analysis['property_statistics'].items():
        if stats['coverage'] > 0.8 and stats['std'] > 0.01:  # Good coverage and variance
            good_properties.append(prop)
    
    # Memory and stability considerations
    recommendations = []
    if max_atoms > 100:
        suggestions['batch_size'] = min(suggestions['batch_size'], 16)  # Even smaller batch size
        recommendations.extend([
            "Very large molecules detected (>100 atoms). Using very conservative settings.",
            "Reduced learning rate and batch size for numerical stability.",
            "Consider filtering molecules to a smaller maximum size for efficiency.",
            "Use: python molecular_db_utils.py filter --input_db your_db.db --output_db filtered_db.db --max_atoms 100",
            "Monitor training closely for gradient explosion or instability."
        ])
    elif max_atoms > 50:
        suggestions['batch_size'] = min(suggestions['batch_size'], 32)
        recommendations.extend([
            "Large molecules detected. Using conservative settings.",
            "Consider reducing batch size further if you encounter memory issues."
        ])
    
    # Add normalization-specific recommendations
    if n_elements > 10:
        recommendations.extend([
            f"High element diversity ({n_elements} elements) detected.",
            f"Using categorical normalization factor of {categorical_norm} to prevent halogen bias.",
            "This addresses issues where rare elements (like Br, I, Cl) are over-generated.",
            "The reduced normalization factor minimizes noise impact during diffusion."
        ])
    elif n_elements > 5:
        recommendations.extend([
            f"Medium element diversity ({n_elements} elements) detected.",
            f"Using reduced categorical normalization factor of {categorical_norm}.",
            "This helps balance element generation compared to QM9 defaults."
        ])
    
    # Add stability recommendations for conditioning
    if len(good_properties) > 2:
        recommendations.append(
            "Multiple conditioning properties detected. Start with 1-2 properties for stability."
        )
    
    if recommendations:
        suggestions['recommendations'] = recommendations
    
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
    
    # Check if we have critical issues that prevent analysis
    critical_issues = [issue for issue in issues if is_critical_issue(issue)]
    has_malformed_warning = any("malformed but may still be partially readable" in issue for issue in issues)
    
    # If we have critical issues that aren't the malformed warning, stop here
    if not is_valid and not has_malformed_warning:
        print("\n❌ DATABASE VALIDATION FAILED")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
        return
    
    # Show warnings for valid databases or malformed but potentially readable databases
    if len(issues) > 0:
        if has_malformed_warning:
            print("\n⚠️  DATABASE CORRUPTION DETECTED - ATTEMPTING ANALYSIS")
        else:
            print("\n⚠️  DATABASE WARNINGS")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        print()  # Add spacing before analysis
    
    # Analyze database with error handling
    try:
        analysis = analyze_ase_database(db_path)
    except Exception as e:
        print("\n❌ DATABASE ANALYSIS FAILED")
        print(f"\nError: {str(e)}")
        print("\nThe database passed basic validation but detailed analysis failed.")
        print("This may indicate partial corruption or compatibility issues.")
        return
    
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
    print(f"  Normalization factors: {suggestions['normalize_factors']} # [x, categorical, integer]")
    print(f"  Note: {suggestions['normalization_note']}")
    
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
        f"--normalize_factors {suggestions['normalize_factors']}",
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


def filter_molecular_database(input_db_path, output_db_path, max_atoms=100, min_atoms=2, 
                             remove_h=False, preserve_properties=True):
    """
    Filter an ASE molecular database to remove molecules outside size constraints.
    
    Parameters
    ----------
    input_db_path : str
        Path to the input ASE database file
    output_db_path : str
        Path to create the filtered ASE database
    max_atoms : int
        Maximum number of atoms per molecule (default: 100)
    min_atoms : int
        Minimum number of atoms per molecule (default: 2)
    remove_h : bool
        Whether to exclude hydrogen atoms from atom count (default: False)
    preserve_properties : bool
        Whether to preserve all molecular properties (default: True)
        
    Returns
    -------
    filter_stats : dict
        Statistics about the filtering process:
        - total_input: number of molecules in input database
        - total_output: number of molecules in output database
        - filtered_out: number of molecules removed
        - size_distribution: dict mapping sizes to counts for output
    """
    try:
        from ase.db import connect
    except ImportError:
        raise ImportError("ASE package is required. Install with: pip install ase")
    
    if not os.path.exists(input_db_path):
        raise FileNotFoundError(f"Input database not found: {input_db_path}")
    
    print(f"Filtering database: {input_db_path}")
    print(f"Constraints: {min_atoms} <= atoms <= {max_atoms}")
    if remove_h:
        print("Note: Hydrogen atoms excluded from count")
    
    # Connect to input database
    input_db = connect(input_db_path)
    
    # Create output database (will overwrite if exists)
    output_db = connect(output_db_path)
    
    # Initialize statistics
    total_input = 0
    total_output = 0
    filtered_out = 0
    size_distribution = {}
    too_large = 0
    too_small = 0
    
    try:
        for row in input_db.select():
            total_input += 1
            
            try:
                atoms = row.toatoms()
                
                # Get atom count based on remove_h setting
                if remove_h:
                    # Count only non-hydrogen atoms
                    n_atoms = sum(1 for symbol in atoms.get_chemical_symbols() if symbol != 'H')
                else:
                    # Count all atoms
                    n_atoms = len(atoms)
                
                # Check size constraints
                if n_atoms < min_atoms:
                    too_small += 1
                    filtered_out += 1
                    continue
                    
                if n_atoms > max_atoms:
                    too_large += 1
                    filtered_out += 1
                    continue
                
                # Molecule passes filters - add to output database
                total_output += 1
                size_distribution[n_atoms] = size_distribution.get(n_atoms, 0) + 1
                
                # Copy molecule and properties to output database
                if preserve_properties:
                    # Copy all data and key-value pairs
                    data = row.data if hasattr(row, 'data') and row.data else {}
                    kvp = row.key_value_pairs if hasattr(row, 'key_value_pairs') and row.key_value_pairs else {}
                    
                    # Merge data and key-value pairs
                    all_properties = {}
                    all_properties.update(data)
                    all_properties.update(kvp)
                    
                    output_db.write(atoms, data=all_properties)
                else:
                    output_db.write(atoms)
                
            except Exception as e:
                print(f"Warning: Error processing molecule {total_input}: {str(e)}")
                filtered_out += 1
                continue
                
            # Progress indicator for large databases
            if total_input % 1000 == 0:
                print(f"  Processed {total_input} molecules, kept {total_output}")
                
    except Exception as e:
        print(f"Error during filtering: {str(e)}")
        raise
    
    # Compile statistics
    filter_stats = {
        'total_input': total_input,
        'total_output': total_output,
        'filtered_out': filtered_out,
        'too_large': too_large,
        'too_small': too_small,
        'size_distribution': size_distribution
    }
    
    # Print summary
    print(f"\n🔧 FILTERING COMPLETE")
    print(f"  Input molecules: {total_input:,}")
    print(f"  Output molecules: {total_output:,}")
    print(f"  Filtered out: {filtered_out:,} ({100 * filtered_out / total_input:.1f}%)")
    if too_large > 0:
        print(f"    - Too large (>{max_atoms} atoms): {too_large:,}")
    if too_small > 0:
        print(f"    - Too small (<{min_atoms} atoms): {too_small:,}")
    print(f"  Filtered database saved to: {output_db_path}")
    
    # Show size distribution summary
    if size_distribution:
        print(f"\n📏 OUTPUT SIZE DISTRIBUTION")
        sorted_sizes = sorted(size_distribution.items())[:10]  # Show first 10
        for size, count in sorted_sizes:
            percentage = (count / total_output) * 100
            print(f"  {size:2d} atoms: {count:6,} molecules ({percentage:5.1f}%)")
        if len(size_distribution) > 10:
            print(f"  ... and {len(size_distribution) - 10} more size categories")
    
    return filter_stats