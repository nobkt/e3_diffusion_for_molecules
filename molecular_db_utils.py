#!/usr/bin/env python3
"""
Utility script for working with general molecular databases in E3 Diffusion.

This script provides easy-to-use commands for:
1. Analyzing molecular databases
2. Creating optimized configurations
3. Validating database compatibility
4. Getting training recommendations

Usage examples:
    python molecular_db_utils.py analyze --db_path molecules.db
    python molecular_db_utils.py create-config --db_path molecules.db --output config.json
    python molecular_db_utils.py validate --db_path molecules.db
    python molecular_db_utils.py recommend --db_path molecules.db
"""

import argparse
import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qm9.general_molecular_db import (
    analyze_ase_database,
    create_optimal_dataset_config,
    validate_molecular_database,
    suggest_training_parameters,
    print_database_summary,
    save_dataset_config
)


def cmd_analyze(args):
    """Analyze a molecular database and print summary."""
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found: {args.db_path}")
        return 1
    
    try:
        print_database_summary(args.db_path, save_summary=args.save_summary)
        return 0
    except Exception as e:
        print(f"Error analyzing database: {e}")
        return 1


def cmd_create_config(args):
    """Create an optimized dataset configuration."""
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found: {args.db_path}")
        return 1
    
    try:
        print(f"Analyzing database: {args.db_path}")
        analysis = analyze_ase_database(args.db_path, remove_h=args.remove_h)
        
        print("Creating optimized configuration...")
        config = create_optimal_dataset_config(
            analysis, 
            dataset_name=args.dataset_name or os.path.splitext(os.path.basename(args.db_path))[0],
            with_h=not args.remove_h
        )
        
        if args.output:
            save_dataset_config(config, args.output)
        else:
            print("\nGenerated configuration:")
            print(json.dumps(config, indent=2))
        
        return 0
    except Exception as e:
        print(f"Error creating configuration: {e}")
        return 1


def cmd_validate(args):
    """Validate database compatibility."""
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found: {args.db_path}")
        return 1
    
    try:
        is_valid, issues, recommendations = validate_molecular_database(args.db_path)
        
        print(f"Database validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        return 0 if is_valid else 1
    except Exception as e:
        print(f"Error validating database: {e}")
        return 1


def cmd_recommend(args):
    """Get training parameter recommendations."""
    if not os.path.exists(args.db_path):
        print(f"Error: Database file not found: {args.db_path}")
        return 1
    
    try:
        analysis = analyze_ase_database(args.db_path, remove_h=args.remove_h)
        suggestions = suggest_training_parameters(analysis)
        
        print("🚀 TRAINING RECOMMENDATIONS")
        print("=" * 50)
        
        print(f"Model Parameters:")
        print(f"  --batch_size {suggestions['batch_size']}")
        print(f"  --n_epochs {suggestions['n_epochs']}")
        print(f"  --nf {suggestions['nf']}")
        print(f"  --n_layers {suggestions['n_layers']}")
        print(f"  --lr {suggestions['lr']}")
        print(f"  --diffusion_steps {suggestions['diffusion_steps']}")
        
        if 'recommended_conditioning' in suggestions:
            print(f"\nRecommended conditioning properties:")
            for prop in suggestions['recommended_conditioning']:
                print(f"  - {prop}")
        
        if 'recommendations' in suggestions:
            print(f"\nAdditional recommendations:")
            for rec in suggestions['recommendations']:
                print(f"  - {rec}")
        
        print(f"\nExample training command:")
        cmd_parts = [
            "python main_qm9.py",
            "--dataset ase_db",
            f"--ase_db_path {args.db_path}",
            f"--batch_size {suggestions['batch_size']}",
            f"--n_epochs {suggestions['n_epochs']}",
            f"--nf {suggestions['nf']}",
            f"--n_layers {suggestions['n_layers']}",
            f"--lr {suggestions['lr']}",
        ]
        
        if args.remove_h:
            cmd_parts.append("--remove_h")
        
        if 'recommended_conditioning' in suggestions:
            cmd_parts.append(f"--conditioning {' '.join(suggestions['recommended_conditioning'])}")
        
        print("  " + " \\\n    ".join(cmd_parts))
        
        return 0
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return 1


def cmd_convert_pubchem(args):
    """Convert PubChem SDF files to ASE database format."""
    try:
        from ase.io import read, write
        from ase.db import connect
    except ImportError:
        print("Error: ASE package is required. Install with: pip install ase")
        return 1
    
    if not os.path.exists(args.sdf_path):
        print(f"Error: SDF file not found: {args.sdf_path}")
        return 1
    
    try:
        print(f"Reading molecules from SDF file: {args.sdf_path}")
        molecules = read(args.sdf_path, index=':')  # Read all molecules
        
        print(f"Found {len(molecules)} molecules")
        
        if args.max_molecules and len(molecules) > args.max_molecules:
            molecules = molecules[:args.max_molecules]
            print(f"Limited to first {args.max_molecules} molecules")
        
        # Create ASE database
        print(f"Creating ASE database: {args.output}")
        db = connect(args.output)
        
        for i, mol in enumerate(molecules):
            # Add basic properties if available
            properties = {}
            if hasattr(mol, 'info') and mol.info:
                properties.update(mol.info)
            
            db.write(mol, data=properties)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(molecules)} molecules")
        
        print(f"Successfully created ASE database with {len(molecules)} molecules")
        
        # Analyze the created database
        if args.analyze:
            print("\nAnalyzing created database...")
            print_database_summary(args.output)
        
        return 0
    except Exception as e:
        print(f"Error converting SDF to ASE database: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Utility for working with general molecular databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a database
  python molecular_db_utils.py analyze --db_path molecules.db
  
  # Create optimized configuration
  python molecular_db_utils.py create-config --db_path molecules.db --output config.json
  
  # Validate database
  python molecular_db_utils.py validate --db_path molecules.db
  
  # Get training recommendations
  python molecular_db_utils.py recommend --db_path molecules.db
  
  # Convert PubChem SDF to ASE database
  python molecular_db_utils.py convert-pubchem --sdf_path pubchem.sdf --output molecules.db
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze molecular database')
    analyze_parser.add_argument('--db_path', required=True, help='Path to ASE database file')
    analyze_parser.add_argument('--save_summary', action='store_true', help='Save summary to file')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create optimized dataset configuration')
    config_parser.add_argument('--db_path', required=True, help='Path to ASE database file')
    config_parser.add_argument('--output', help='Output configuration file path (JSON)')
    config_parser.add_argument('--dataset_name', help='Name for the dataset configuration')
    config_parser.add_argument('--remove_h', action='store_true', help='Remove hydrogen atoms')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate database compatibility')
    validate_parser.add_argument('--db_path', required=True, help='Path to ASE database file')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Get training recommendations')
    recommend_parser.add_argument('--db_path', required=True, help='Path to ASE database file')
    recommend_parser.add_argument('--remove_h', action='store_true', help='Remove hydrogen atoms')
    
    # Convert PubChem command
    convert_parser = subparsers.add_parser('convert-pubchem', help='Convert PubChem SDF to ASE database')
    convert_parser.add_argument('--sdf_path', required=True, help='Path to PubChem SDF file')
    convert_parser.add_argument('--output', required=True, help='Output ASE database path')
    convert_parser.add_argument('--max_molecules', type=int, help='Maximum number of molecules to convert')
    convert_parser.add_argument('--analyze', action='store_true', help='Analyze the created database')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # Call the appropriate command function
    command_map = {
        'analyze': cmd_analyze,
        'create-config': cmd_create_config,
        'validate': cmd_validate,
        'recommend': cmd_recommend,
        'convert-pubchem': cmd_convert_pubchem
    }
    
    return command_map[args.command](args)


if __name__ == '__main__':
    sys.exit(main())