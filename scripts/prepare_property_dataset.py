"""
Data Preparation Script for Property-Conditioned Crystal Generation

This script prepares and validates crystal datasets with property values,
implementing Phase 3 Component P3-3 of the property-conditioned molecular 
crystal generation system.

Key Features:
- Merge property CSV with crystal database
- Validate data completeness
- Check for zero-variance properties
- Compute and display statistics
- Create train/val/test splits

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Strict validation throughout
- Clear error messages for data issues
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from ase.db import connect
from typing import List, Dict


parser = argparse.ArgumentParser(description='Prepare Property-Conditioned Crystal Dataset')

parser.add_argument('--crystal_db', type=str, required=True,
                    help='Path to input crystal ASE database')
parser.add_argument('--property_csv', type=str, required=True,
                    help='Path to CSV file with property data')
parser.add_argument('--output_db', type=str, required=True,
                    help='Path to output crystal database with properties')
parser.add_argument('--property_names', nargs='+', type=str, required=True,
                    help='List of property names to include')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='Training set ratio')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='Validation set ratio')
parser.add_argument('--test_ratio', type=float, default=0.1,
                    help='Test set ratio')
parser.add_argument('--validate', action='store_true',
                    help='Run validation checks before merging')
parser.add_argument('--crystal_id_column', type=str, default='crystal_id',
                    help='Name of crystal ID column in CSV')

args = parser.parse_args()


def validate_properties(db, properties_df: pd.DataFrame, property_names: List[str],
                       crystal_id_column: str) -> None:
    """
    Validate property data completeness and variance.
    
    Args:
        db: ASE database connection
        properties_df: DataFrame with property data
        property_names: List of property names to validate
        crystal_id_column: Name of crystal ID column
    
    Raises:
        ValueError: If validation fails
    """
    print("\nRunning validation checks...")
    
    # Check crystal ID column exists
    if crystal_id_column not in properties_df.columns:
        raise ValueError(
            f"Crystal ID column '{crystal_id_column}' not found in CSV. "
            f"Available columns: {list(properties_df.columns)}"
        )
    
    # Check all property columns exist
    missing_props = [p for p in property_names if p not in properties_df.columns]
    if missing_props:
        raise ValueError(
            f"Properties not found in CSV: {missing_props}. "
            f"Available columns: {list(properties_df.columns)}"
        )
    
    # Get all crystal IDs from database
    crystal_ids_db = set()
    for row in db.select():
        # Try to get crystal_id from data, fallback to row.id
        crystal_id = row.data.get('crystal_id', row.id)
        crystal_ids_db.add(crystal_id)
    
    # Get crystal IDs from CSV
    crystal_ids_csv = set(properties_df[crystal_id_column].values)
    
    # Check for missing crystals in CSV
    missing_in_csv = crystal_ids_db - crystal_ids_csv
    if missing_in_csv:
        print(f"WARNING: {len(missing_in_csv)} crystals in database have no properties in CSV")
        if len(missing_in_csv) <= 10:
            print(f"  Missing IDs: {list(missing_in_csv)[:10]}")
        raise ValueError(
            f"Properties missing for {len(missing_in_csv)} crystals. "
            f"All crystals must have property data."
        )
    
    # Check for extra crystals in CSV (warning only)
    extra_in_csv = crystal_ids_csv - crystal_ids_db
    if extra_in_csv:
        print(f"WARNING: {len(extra_in_csv)} property entries have no matching crystal in database")
        if len(extra_in_csv) <= 10:
            print(f"  Extra IDs: {list(extra_in_csv)[:10]}")
    
    # Check property variance
    print("\nProperty statistics:")
    zero_variance_props = []
    for prop_name in property_names:
        values = properties_df[prop_name].values
        
        # Check for NaN values
        n_nan = np.isnan(values).sum()
        if n_nan > 0:
            raise ValueError(f"Property '{prop_name}' has {n_nan} NaN values")
        
        # Compute statistics
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"  {prop_name}:")
        print(f"    Mean: {mean:.4f}")
        print(f"    Std:  {std:.4f}")
        print(f"    Min:  {min_val:.4f}")
        print(f"    Max:  {max_val:.4f}")
        
        # Check variance
        if std < 1e-8:
            zero_variance_props.append(prop_name)
    
    if zero_variance_props:
        raise ValueError(
            f"Properties with zero variance: {zero_variance_props}. "
            f"These properties cannot be used for conditioning."
        )
    
    print("\n✓ All validation checks passed")


def merge_properties(input_db_path: str, properties_df: pd.DataFrame, 
                     output_db_path: str, property_names: List[str],
                     crystal_id_column: str) -> None:
    """
    Merge properties from CSV into crystal database.
    
    Args:
        input_db_path: Path to input database
        properties_df: DataFrame with property data
        output_db_path: Path to output database
        property_names: List of property names to merge
        crystal_id_column: Name of crystal ID column
    """
    print("\nMerging properties into database...")
    
    # Open databases
    input_db = connect(input_db_path)
    output_db = connect(output_db_path)
    
    # Process each crystal
    n_processed = 0
    n_skipped = 0
    
    for row in input_db.select():
        # Get crystal ID
        crystal_id = row.data.get('crystal_id', row.id)
        
        # Find matching properties
        prop_rows = properties_df[properties_df[crystal_id_column] == crystal_id]
        
        if len(prop_rows) == 0:
            print(f"WARNING: No properties found for crystal {crystal_id}, skipping")
            n_skipped += 1
            continue
        
        if len(prop_rows) > 1:
            print(f"WARNING: Multiple property entries for crystal {crystal_id}, using first")
        
        # Get property values
        prop_row = prop_rows.iloc[0]
        
        # Copy data and add properties
        data = dict(row.data) if row.data else {}
        for prop_name in property_names:
            if prop_name not in prop_row or pd.isna(prop_row[prop_name]):
                raise ValueError(f"Property {prop_name} missing or NaN for crystal {crystal_id}")
            data[prop_name] = float(prop_row[prop_name])
        
        # Ensure crystal_id is in data
        data['crystal_id'] = crystal_id
        
        # Write to output database
        atoms = row.toatoms()
        output_db.write(atoms, data=data)
        
        n_processed += 1
        
        if n_processed % 100 == 0:
            print(f"  Processed {n_processed} crystals...")
    
    input_db._close()
    output_db._close()
    
    print(f"\n✓ Merged properties for {n_processed} crystals")
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} crystals without properties")


def compute_and_display_statistics(db_path: str, property_names: List[str]) -> Dict:
    """
    Compute and display property statistics from database.
    
    Args:
        db_path: Path to database
        property_names: List of property names
    
    Returns:
        dict: Statistics dictionary
    """
    print("\nComputing final statistics...")
    
    db = connect(db_path)
    
    # Collect property values
    properties = {name: [] for name in property_names}
    
    for row in db.select():
        for prop_name in property_names:
            if prop_name in row.data:
                properties[prop_name].append(row.data[prop_name])
    
    db._close()
    
    # Compute statistics
    stats = {}
    print("\nFinal property statistics:")
    for prop_name in property_names:
        values = np.array(properties[prop_name])
        
        stats[prop_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'n_samples': len(values)
        }
        
        print(f"  {prop_name}:")
        print(f"    Mean: {stats[prop_name]['mean']:.4f}")
        print(f"    Std:  {stats[prop_name]['std']:.4f}")
        print(f"    Min:  {stats[prop_name]['min']:.4f}")
        print(f"    Max:  {stats[prop_name]['max']:.4f}")
        print(f"    N:    {stats[prop_name]['n_samples']}")
    
    return stats


def create_splits(db_path: str, train_ratio: float, val_ratio: float, 
                 test_ratio: float, output_dir: Path) -> None:
    """
    Create train/val/test splits and save indices.
    
    Args:
        db_path: Path to database
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        output_dir: Output directory for split files
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    print("\nCreating train/val/test splits...")
    
    # Get all indices
    db = connect(db_path)
    all_indices = [row.id for row in db.select()]
    n_total = len(all_indices)
    db._close()
    
    # Shuffle indices
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    indices = np.array(all_indices)
    rng.shuffle(indices)
    
    # Compute split sizes
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Create splits
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()
    
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val:   {len(val_indices)} samples")
    print(f"  Test:  {len(test_indices)} samples")
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    splits_path = output_dir / 'dataset_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Splits saved to {splits_path}")


def main():
    print("="*70)
    print("Property-Conditioned Crystal Dataset Preparation")
    print("="*70)
    
    # Validate arguments
    if not Path(args.crystal_db).exists():
        raise FileNotFoundError(f"Crystal database not found: {args.crystal_db}")
    
    if not Path(args.property_csv).exists():
        raise FileNotFoundError(f"Property CSV not found: {args.property_csv}")
    
    # Check output database doesn't exist
    if Path(args.output_db).exists():
        raise FileExistsError(
            f"Output database already exists: {args.output_db}. "
            f"Please remove it or choose a different output path."
        )
    
    print(f"\nInput crystal database: {args.crystal_db}")
    print(f"Property CSV: {args.property_csv}")
    print(f"Output database: {args.output_db}")
    print(f"Properties: {args.property_names}")
    
    # Load property CSV
    print("\nLoading property CSV...")
    properties_df = pd.read_csv(args.property_csv)
    print(f"  Loaded {len(properties_df)} property entries")
    print(f"  Columns: {list(properties_df.columns)}")
    
    # Open input database
    input_db = connect(args.crystal_db)
    n_crystals = len([row for row in input_db.select()])
    print(f"\nInput database contains {n_crystals} crystals")
    
    # Validate if requested
    if args.validate:
        validate_properties(input_db, properties_df, args.property_names, args.crystal_id_column)
    
    input_db._close()
    
    # Merge properties
    merge_properties(
        args.crystal_db,
        properties_df,
        args.output_db,
        args.property_names,
        args.crystal_id_column
    )
    
    # Compute and display statistics
    stats = compute_and_display_statistics(args.output_db, args.property_names)
    
    # Save statistics to file
    output_dir = Path(args.output_db).parent
    stats_path = output_dir / 'property_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Statistics saved to {stats_path}")
    
    # Create splits
    create_splits(
        args.output_db,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        output_dir
    )
    
    print("\n" + "="*70)
    print("Dataset preparation complete!")
    print(f"Output database: {args.output_db}")
    print(f"Statistics: {stats_path}")
    print(f"Splits: {output_dir / 'dataset_splits.json'}")
    print("="*70)


if __name__ == '__main__':
    main()
