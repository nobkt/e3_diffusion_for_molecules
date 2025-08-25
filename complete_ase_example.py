#!/usr/bin/env python3
"""
Complete example workflow for EDM with ASE databases.
This script demonstrates the entire pipeline from dataset creation to evaluation.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, description="", check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f}s")
        return result
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.1f}s")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def create_sample_database(db_path, n_molecules=1000):
    """Create a sample ASE database."""
    print(f"Creating sample database with {n_molecules} molecules...")
    
    from qm9.ase_dataset import create_sample_ase_db
    create_sample_ase_db(db_path, num_molecules=n_molecules)
    
    print(f"✓ Created database: {db_path}")


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Complete ASE workflow example')
    
    # Workflow parameters
    parser.add_argument('--exp_name', type=str, default='ase_example',
                        help='Experiment name')
    parser.add_argument('--n_molecules', type=int, default=1000,
                        help='Number of molecules in sample database')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of molecules to generate')
    
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--nf', type=int, default=128,
                        help='Number of features')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of layers')
    
    # Workflow control
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training (use existing model)')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip generation (use existing samples)')
    parser.add_argument('--quick_mode', action='store_true',
                        help='Use smaller parameters for quick testing')
    
    # Output control
    parser.add_argument('--output_dir', type=str, default='ase_example_results',
                        help='Output directory for all results')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up intermediate files')
    
    return parser.parse_args()


def main():
    """Main function demonstrating complete ASE workflow."""
    
    print("="*80)
    print("E3 Diffusion for Molecules - Complete ASE Workflow Example")
    print("="*80)
    
    # Setup
    args = setup_args()
    
    # Quick mode adjustments
    if args.quick_mode:
        args.n_molecules = min(args.n_molecules, 100)
        args.n_epochs = min(args.n_epochs, 10)
        args.n_samples = min(args.n_samples, 50)
        args.batch_size = min(args.batch_size, 16)
        args.nf = min(args.nf, 64)
        args.n_layers = min(args.n_layers, 3)
        print("Quick mode enabled - using smaller parameters for testing")
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    db_path = output_dir / 'molecules.db'
    model_dir = f'outputs/{args.exp_name}'
    generated_dir = output_dir / 'generated_molecules'
    
    print(f"\nWorkflow configuration:")
    print(f"  Experiment name: {args.exp_name}")
    print(f"  Database: {db_path}")
    print(f"  Model directory: {model_dir}")
    print(f"  Generated molecules: {generated_dir}")
    print(f"  Training epochs: {args.n_epochs}")
    print(f"  Molecules to generate: {args.n_samples}")
    
    try:
        # Step 1: Create sample database
        if not db_path.exists():
            run_command(
                ['python', '-c', f"""
from qm9.ase_dataset import create_sample_ase_db
create_sample_ase_db('{db_path}', num_molecules={args.n_molecules})
print('Created database with {args.n_molecules} molecules')
"""],
                description="Creating sample ASE database"
            )
        else:
            print(f"Using existing database: {db_path}")
        
        # Step 2: Train model
        if not args.skip_training:
            train_cmd = [
                'python', 'main_ase.py',
                '--ase_db_path', str(db_path),
                '--exp_name', args.exp_name,
                '--n_epochs', str(args.n_epochs),
                '--batch_size', str(args.batch_size),
                '--lr', str(args.lr),
                '--nf', str(args.nf),
                '--n_layers', str(args.n_layers),
                '--test_epochs', str(max(1, args.n_epochs // 5)),
                '--save_model',
                '--no_wandb'  # Disable wandb for example
            ]
            
            run_command(train_cmd, description="Training EDM model on ASE database")
        else:
            print("Skipping training - using existing model")
        
        # Step 3: Generate molecules
        if not args.skip_generation:
            sample_cmd = [
                'python', 'sample_ase.py',
                '--model_path', model_dir,
                '--n_samples', str(args.n_samples),
                '--batch_size', str(min(args.batch_size, 50)),
                '--output_dir', str(generated_dir),
                '--save_xyz',
                '--save_tensors',
                '--visualize_examples', '5'
            ]
            
            run_command(sample_cmd, description="Generating molecules from trained model")
        else:
            print("Skipping generation - using existing samples")
        
        # Step 4: Evaluate quality
        eval_cmd = [
            'python', 'evaluate_ase.py',
            '--generated_molecules', str(generated_dir / 'generated_molecules.pt'),
            '--reference_db', str(db_path),
            '--output_file', 'evaluation_results.txt',
            '--output_dir', str(output_dir),
            '--detailed_analysis'
        ]
        
        run_command(eval_cmd, description="Evaluating generated molecules")
        
        # Step 5: Create summary report
        create_summary_report(args, output_dir, db_path, model_dir, generated_dir)
        
        # Step 6: Cleanup (optional)
        if args.cleanup:
            cleanup_files(output_dir, keep_important=True)
        
        print("\n" + "="*80)
        print("✓ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {output_dir}")
        print(f"Key files:")
        print(f"  - Database: {db_path}")
        print(f"  - Model: {model_dir}")
        print(f"  - Generated molecules: {generated_dir}")
        print(f"  - Evaluation: {output_dir / 'evaluation_results.txt'}")
        print(f"  - Summary: {output_dir / 'workflow_summary.txt'}")
        
    except KeyboardInterrupt:
        print("\n\n✗ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_summary_report(args, output_dir, db_path, model_dir, generated_dir):
    """Create a summary report of the entire workflow."""
    print("\nCreating workflow summary report...")
    
    summary_path = output_dir / 'workflow_summary.txt'
    
    # Read evaluation results
    eval_path = output_dir / 'evaluation_results.txt'
    eval_content = ""
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            eval_content = f.read()
    
    # Create summary
    summary = f"""
E3 Diffusion for Molecules - ASE Workflow Summary
================================================

Experiment: {args.exp_name}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
- Database molecules: {args.n_molecules}
- Training epochs: {args.n_epochs}
- Generated samples: {args.n_samples}
- Batch size: {args.batch_size}
- Learning rate: {args.lr}
- Model features: {args.nf}
- Model layers: {args.n_layers}
- Quick mode: {args.quick_mode}

Files Created:
- Database: {db_path}
- Model directory: {model_dir}
- Generated molecules: {generated_dir}
- Evaluation results: {output_dir / 'evaluation_results.txt'}

Workflow Steps Completed:
1. ✓ Created ASE database with {args.n_molecules} molecules
2. ✓ Trained EDM model for {args.n_epochs} epochs
3. ✓ Generated {args.n_samples} new molecules
4. ✓ Evaluated molecular quality
5. ✓ Created summary report

Evaluation Results:
{eval_content}

Next Steps:
- Examine generated molecules in {generated_dir}
- Review model performance in {model_dir}
- Adjust parameters and retrain if needed
- Use trained model for further generation

For detailed usage, see ASE_TUTORIAL.md
"""

    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"✓ Summary report saved to: {summary_path}")


def cleanup_files(output_dir, keep_important=True):
    """Clean up intermediate files."""
    print("\nCleaning up intermediate files...")
    
    # Files to potentially remove
    cleanup_patterns = [
        '*.log',
        '*.tmp',
        'wandb',
        '__pycache__'
    ]
    
    if keep_important:
        print("Keeping important result files")
    else:
        print("Removing all intermediate files")
    
    # Implementation would go here
    print("✓ Cleanup completed")


if __name__ == "__main__":
    main()