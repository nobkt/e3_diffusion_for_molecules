"""
Validate Generated Crystals using Property Predictor

Validates generated crystals by predicting their properties and comparing
against target property values.

Usage:
    python validate_generated_crystals.py \\
        --predictor_path outputs/property_predictor/checkpoint_best.pt \\
        --crystal_dir generated_samples/ \\
        --target_bandgap 2.5 \\
        --target_melting_point 180.0 \\
        --output_report validation_report.txt

This implements Component P4-1 (Property Validation System) from Phase 4.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

import torch
import numpy as np
from ase.io import read as ase_read

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from crystal.evaluation.property_predictor import PropertyPredictor


class CrystalValidator:
    """
    Validates generated crystals using property predictor.
    
    Computes prediction errors and generates validation reports.
    """
    
    def __init__(
        self,
        predictor: PropertyPredictor,
        property_names: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.predictor = predictor.to(device)
        self.predictor.eval()
        self.property_names = property_names
        self.device = device
    
    def load_crystal_from_cif(self, cif_path: str) -> Dict[str, torch.Tensor]:
        """
        Load crystal structure from CIF file.
        
        Args:
            cif_path: Path to CIF file
            
        Returns:
            crystal_data: Dictionary with positions, cell, and atomic_numbers
        """
        # Read CIF file using ASE
        atoms = ase_read(cif_path)
        
        # Extract data
        positions = torch.tensor(
            atoms.get_scaled_positions(),  # Fractional coordinates
            dtype=torch.float32
        )
        cell = torch.tensor(
            atoms.get_cell().array,
            dtype=torch.float32
        )
        atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(),
            dtype=torch.long
        )
        
        # Add batch dimension
        positions = positions.unsqueeze(0)  # [1, n_atoms, 3]
        cell = cell.unsqueeze(0)  # [1, 3, 3]
        atomic_numbers = atomic_numbers.unsqueeze(0)  # [1, n_atoms]
        
        return {
            'positions': positions,
            'cell': cell,
            'atomic_numbers': atomic_numbers,
        }
    
    def predict_properties(
        self,
        crystal_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Predict properties for a crystal structure.
        
        Args:
            crystal_data: Dictionary with positions, cell, and atomic_numbers
            
        Returns:
            predictions: Dictionary mapping property names to predicted values
        """
        # Move to device
        positions = crystal_data['positions'].to(self.device)
        cell = crystal_data['cell'].to(self.device)
        atomic_numbers = crystal_data['atomic_numbers'].to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions_tensor = self.predictor(
                positions, cell, atomic_numbers,
                return_normalized=False
            )
        
        # Convert to Python floats
        predictions = {
            name: pred.cpu().item()
            for name, pred in predictions_tensor.items()
        }
        
        return predictions
    
    def validate_crystal(
        self,
        cif_path: str,
        target_properties: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Validate a single crystal structure.
        
        Args:
            cif_path: Path to CIF file
            target_properties: Dictionary of target property values
            
        Returns:
            validation_result: Dictionary with predictions, errors, and statistics
        """
        # Load crystal
        crystal_data = self.load_crystal_from_cif(cif_path)
        
        # Predict properties
        predictions = self.predict_properties(crystal_data)
        
        # Compute errors
        errors = {}
        relative_errors = {}
        
        for prop_name in self.property_names:
            if prop_name in target_properties:
                target = target_properties[prop_name]
                pred = predictions[prop_name]
                
                # Absolute error
                error = abs(pred - target)
                errors[prop_name] = error
                
                # Relative error (percentage)
                if abs(target) > 1e-8:
                    rel_error = (error / abs(target)) * 100
                    relative_errors[prop_name] = rel_error
        
        return {
            'crystal_path': str(cif_path),
            'predictions': predictions,
            'targets': target_properties,
            'absolute_errors': errors,
            'relative_errors': relative_errors,
        }
    
    def validate_directory(
        self,
        crystal_dir: str,
        target_properties: Dict[str, float],
        pattern: str = '*.cif'
    ) -> List[Dict[str, any]]:
        """
        Validate all crystals in a directory.
        
        Args:
            crystal_dir: Directory containing CIF files
            target_properties: Dictionary of target property values
            pattern: File pattern to match (default: *.cif)
            
        Returns:
            results: List of validation results
        """
        crystal_dir = Path(crystal_dir)
        cif_files = sorted(crystal_dir.glob(pattern))
        
        if not cif_files:
            raise ValueError(f'No CIF files found in {crystal_dir} matching {pattern}')
        
        print(f'Validating {len(cif_files)} crystals...')
        
        results = []
        for cif_path in cif_files:
            try:
                result = self.validate_crystal(cif_path, target_properties)
                results.append(result)
            except Exception as e:
                print(f'Error validating {cif_path}: {e}')
                continue
        
        return results
    
    def compute_summary_statistics(
        self,
        results: List[Dict[str, any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics across all validated crystals.
        
        Args:
            results: List of validation results
            
        Returns:
            statistics: Dictionary of statistics for each property
        """
        statistics = {}
        
        for prop_name in self.property_names:
            # Collect errors
            abs_errors = [
                r['absolute_errors'][prop_name]
                for r in results
                if prop_name in r['absolute_errors']
            ]
            
            rel_errors = [
                r['relative_errors'][prop_name]
                for r in results
                if prop_name in r['relative_errors']
            ]
            
            if abs_errors:
                statistics[prop_name] = {
                    'mae': np.mean(abs_errors),
                    'std': np.std(abs_errors),
                    'min': np.min(abs_errors),
                    'max': np.max(abs_errors),
                    'mean_relative_error_%': np.mean(rel_errors) if rel_errors else None,
                }
        
        return statistics
    
    def generate_report(
        self,
        results: List[Dict[str, any]],
        output_path: str
    ):
        """
        Generate validation report.
        
        Args:
            results: List of validation results
            output_path: Path to save report
        """
        # Compute statistics
        statistics = self.compute_summary_statistics(results)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('=' * 80 + '\n')
            f.write('CRYSTAL PROPERTY VALIDATION REPORT\n')
            f.write('=' * 80 + '\n\n')
            
            f.write(f'Total crystals validated: {len(results)}\n')
            f.write(f'Properties: {", ".join(self.property_names)}\n\n')
            
            f.write('-' * 80 + '\n')
            f.write('SUMMARY STATISTICS\n')
            f.write('-' * 80 + '\n\n')
            
            for prop_name, stats in statistics.items():
                f.write(f'{prop_name}:\n')
                f.write(f'  Mean Absolute Error (MAE): {stats["mae"]:.4f}\n')
                f.write(f'  Standard Deviation: {stats["std"]:.4f}\n')
                f.write(f'  Min Error: {stats["min"]:.4f}\n')
                f.write(f'  Max Error: {stats["max"]:.4f}\n')
                if stats['mean_relative_error_%'] is not None:
                    f.write(f'  Mean Relative Error: {stats["mean_relative_error_%"]:.2f}%\n')
                f.write('\n')
            
            f.write('-' * 80 + '\n')
            f.write('DETAILED RESULTS\n')
            f.write('-' * 80 + '\n\n')
            
            for i, result in enumerate(results, 1):
                f.write(f'Crystal {i}: {Path(result["crystal_path"]).name}\n')
                
                for prop_name in self.property_names:
                    if prop_name in result['predictions']:
                        pred = result['predictions'][prop_name]
                        target = result['targets'].get(prop_name, None)
                        
                        f.write(f'  {prop_name}:\n')
                        f.write(f'    Predicted: {pred:.4f}\n')
                        if target is not None:
                            f.write(f'    Target: {target:.4f}\n')
                            if prop_name in result['absolute_errors']:
                                abs_err = result['absolute_errors'][prop_name]
                                f.write(f'    Absolute Error: {abs_err:.4f}\n')
                            if prop_name in result['relative_errors']:
                                rel_err = result['relative_errors'][prop_name]
                                f.write(f'    Relative Error: {rel_err:.2f}%\n')
                
                f.write('\n')
            
            f.write('=' * 80 + '\n')
        
        print(f'Validation report saved to {output_path}')


def load_predictor_from_checkpoint(checkpoint_path: str) -> PropertyPredictor:
    """
    Load property predictor from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        predictor: Loaded PropertyPredictor model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract configuration
    property_names = checkpoint['property_names']
    model_config = checkpoint['model_config']
    
    # Create model
    predictor = PropertyPredictor(
        property_names=property_names,
        **model_config
    )
    
    # Load state dict
    predictor.load_state_dict(checkpoint['model_state_dict'])
    
    # Set normalization parameters
    predictor.set_normalization_params(
        checkpoint['property_mean'],
        checkpoint['property_std']
    )
    
    return predictor


def main():
    parser = argparse.ArgumentParser(
        description='Validate generated crystals using property predictor'
    )
    
    # Model arguments
    parser.add_argument(
        '--predictor_path',
        type=str,
        required=True,
        help='Path to property predictor checkpoint'
    )
    
    # Data arguments
    parser.add_argument(
        '--crystal_dir',
        type=str,
        required=True,
        help='Directory containing generated crystal CIF files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.cif',
        help='File pattern to match (default: *.cif)'
    )
    
    # Target properties (parsed dynamically from checkpoint)
    # These will be added as --target_<property> arguments
    
    # Output arguments
    parser.add_argument(
        '--output_report',
        type=str,
        default='validation_report.txt',
        help='Path to save validation report (default: validation_report.txt)'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='Path to save detailed results as JSON (optional)'
    )
    
    # Parse known args first to get predictor path
    args, unknown = parser.parse_known_args()
    
    # Load checkpoint to get property names
    if not os.path.exists(args.predictor_path):
        raise ValueError(f'Predictor checkpoint not found: {args.predictor_path}')
    
    checkpoint = torch.load(args.predictor_path, map_location='cpu')
    property_names = checkpoint['property_names']
    
    # Add target property arguments
    for prop_name in property_names:
        parser.add_argument(
            f'--target_{prop_name}',
            type=float,
            default=None,
            help=f'Target value for {prop_name}'
        )
    
    # Re-parse with property-specific arguments
    args = parser.parse_args()
    
    # Collect target properties
    target_properties = {}
    for prop_name in property_names:
        value = getattr(args, f'target_{prop_name}', None)
        if value is not None:
            target_properties[prop_name] = value
    
    if not target_properties:
        print('Warning: No target properties specified. Only predictions will be shown.')
    
    print('Loading property predictor...')
    predictor = load_predictor_from_checkpoint(args.predictor_path)
    print(f'Properties: {property_names}')
    
    # Create validator
    validator = CrystalValidator(
        predictor=predictor,
        property_names=property_names,
    )
    
    # Validate crystals
    results = validator.validate_directory(
        crystal_dir=args.crystal_dir,
        target_properties=target_properties,
        pattern=args.pattern
    )
    
    if not results:
        print('No crystals were successfully validated.')
        return
    
    print(f'Successfully validated {len(results)} crystals.')
    
    # Generate report
    validator.generate_report(results, args.output_report)
    
    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Detailed results saved to {args.output_json}')
    
    # Print summary to console
    statistics = validator.compute_summary_statistics(results)
    print('\nValidation Summary:')
    for prop_name, stats in statistics.items():
        print(f'{prop_name}:')
        print(f'  MAE: {stats["mae"]:.4f}')
        if stats['mean_relative_error_%'] is not None:
            print(f'  Mean Relative Error: {stats["mean_relative_error_%"]:.2f}%')


if __name__ == '__main__':
    main()
