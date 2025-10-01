#!/usr/bin/env python3
"""
Example script showing how to use the CSV export feature.

This script demonstrates:
1. Loading a dataset (ASE or QM9)
2. Exporting generation conditions to CSV files
3. Reading the generated CSV files

Usage:
    python csv_export_example.py --dataset ase_db --ase_db_path ase.db --output ./csv_output
    python csv_export_example.py --dataset qm9 --datadir qm9/temp --output ./qm9_output
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='CSV Export Example')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['ase_db', 'qm9'],
                       help='Dataset type')
    parser.add_argument('--ase_db_path', type=str, default='ase.db',
                       help='Path to ASE database file')
    parser.add_argument('--datadir', type=str, default='qm9/temp',
                       help='QM9 data directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for CSV files')
    parser.add_argument('--remove_h', action='store_true',
                       help='Remove hydrogen atoms')
    args = parser.parse_args()
    
    # Build the command to run main_qm9.py
    cmd_parts = [
        sys.executable,
        'main_qm9.py',
        '--dataset', args.dataset,
        '--export_conditions_csv', args.output,
        '--no_wandb',
        '--batch_size', '2'
    ]
    
    if args.dataset == 'ase_db':
        cmd_parts.extend(['--ase_db_path', args.ase_db_path])
    else:
        cmd_parts.extend(['--datadir', args.datadir])
    
    if args.remove_h:
        cmd_parts.append('--remove_h')
    
    print("="*60)
    print("CSV Export Example")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    print(f"Output directory: {args.output}")
    print(f"\nCommand: {' '.join(cmd_parts)}")
    print("\n" + "="*60)
    print("Executing...\n")
    
    # Run the command
    import subprocess
    result = subprocess.run(cmd_parts, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"\n✗ Export failed with return code {result.returncode}")
        return 1
    
    # Check and display results
    print("\n" + "="*60)
    print("Checking generated files...")
    print("="*60)
    
    expected_files = [
        'molecular_weight.csv',
        'pi_conjugation_ratio.csv',
        'atom_types_encoding.csv',
        'functional_groups_encoding.csv'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(args.output, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"\n✓ {filename} ({size} bytes)")
            
            # Show first few lines
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:3]
            print(f"  First {len(lines)} lines:")
            for line in lines:
                print(f"    {line.rstrip()}")
        else:
            print(f"\n✗ {filename} - NOT FOUND")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    print(f"\nGenerated CSV files are in: {args.output}")
    print("\nYou can now:")
    print("  1. Open the CSV files in Excel or any spreadsheet software")
    print("  2. Use pandas to analyze the data:")
    print("     import pandas as pd")
    print(f"     df = pd.read_csv('{os.path.join(args.output, 'molecular_weight.csv')}')")
    print("     print(df.head())")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
