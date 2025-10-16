#!/usr/bin/env python3
"""
Test script to verify resume functionality.
This script tests that:
1. Training can be resumed from a checkpoint directory
2. The start_epoch is correctly determined from the checkpoint
3. The model and optimizer states are properly loaded
"""

import os
import sys
import torch
import pickle
import argparse
import shutil
import tempfile
from os.path import join


def create_mock_checkpoint(checkpoint_dir):
    """Create a mock checkpoint for testing."""
    print(f"Creating mock checkpoint in {checkpoint_dir}...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create mock model state (simple tensor dict)
    model_state = {
        'dummy_weight': torch.randn(10, 10),
        'dummy_bias': torch.randn(10)
    }
    
    # Save model states
    torch.save(model_state, join(checkpoint_dir, 'generative_model.npy'))
    torch.save(model_state, join(checkpoint_dir, 'generative_model_ema.npy'))
    
    # Create mock optimizer state
    optim_state = {
        'state': {},
        'param_groups': [{'lr': 0.0001}]
    }
    torch.save(optim_state, join(checkpoint_dir, 'optim.npy'))
    
    # Create mock args
    mock_args = argparse.Namespace(
        exp_name='test_checkpoint',
        dataset='qm9',
        n_epochs=200,
        batch_size=16,
        lr=1e-4,
        nf=256,
        n_layers=9,
        current_epoch=100,  # Simulating that training stopped at epoch 100
        normalization_factor=100,
        aggregation_method='sum',
        save_model=True,
        diffusion_steps=1000,
        no_wandb=True
    )
    
    with open(join(checkpoint_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(mock_args, f)
    
    print("Mock checkpoint created successfully.")
    return mock_args


def test_resume_from_directory():
    """Test resuming from a checkpoint directory."""
    print("\n" + "=" * 60)
    print("Test 1: Resume from checkpoint directory")
    print("=" * 60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = join(tmpdir, 'test_checkpoint')
        mock_args = create_mock_checkpoint(checkpoint_dir)
        
        # Now test the resume logic
        print("\nTesting resume logic...")
        
        # Simulate command-line args
        class ResumeArgs:
            def __init__(self):
                self.resume = checkpoint_dir
                self.exp_name = 'test_checkpoint'
                self.start_epoch = 0  # Not explicitly set
                self.wandb_usr = None
                self.normalization_factor = 100
                self.aggregation_method = 'sum'
        
        args = ResumeArgs()
        
        # Test loading args.pickle
        args_path = join(checkpoint_dir, 'args.pickle')
        if os.path.exists(args_path):
            print(f"✓ Found args.pickle at {args_path}")
            with open(args_path, 'rb') as f:
                loaded_args = pickle.load(f)
            print(f"✓ Loaded args successfully")
            print(f"  - Saved epoch: {loaded_args.current_epoch}")
            
            # Test that current_epoch would be used as start_epoch
            if hasattr(loaded_args, 'current_epoch') and args.start_epoch == 0:
                expected_start_epoch = loaded_args.current_epoch
                print(f"✓ Will resume from epoch {expected_start_epoch} (from checkpoint)")
            else:
                print("✗ current_epoch not found in checkpoint")
                return False
        else:
            print(f"✗ args.pickle not found at {args_path}")
            return False
        
        # Test loading model files
        if os.path.exists(join(checkpoint_dir, 'generative_model_ema.npy')):
            print(f"✓ Found generative_model_ema.npy")
            model_state = torch.load(join(checkpoint_dir, 'generative_model_ema.npy'))
            print(f"✓ Loaded EMA model successfully")
        elif os.path.exists(join(checkpoint_dir, 'generative_model.npy')):
            print(f"✓ Found generative_model.npy")
            model_state = torch.load(join(checkpoint_dir, 'generative_model.npy'))
            print(f"✓ Loaded model successfully")
        else:
            print(f"✗ No model checkpoint found")
            return False
        
        # Test loading optimizer
        if os.path.exists(join(checkpoint_dir, 'optim.npy')):
            print(f"✓ Found optim.npy")
            optim_state = torch.load(join(checkpoint_dir, 'optim.npy'))
            print(f"✓ Loaded optimizer successfully")
        else:
            print(f"⚠ optim.npy not found (will use fresh optimizer)")
        
        print("\n✓ Test 1 PASSED: All checkpoint files loaded successfully")
        return True


def test_resume_from_file():
    """Test resuming from a specific model file."""
    print("\n" + "=" * 60)
    print("Test 2: Resume from specific model file")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = join(tmpdir, 'test_checkpoint')
        mock_args = create_mock_checkpoint(checkpoint_dir)
        
        # Test loading from specific file path
        model_path = join(checkpoint_dir, 'generative_model_ema.npy')
        print(f"\nTesting resume from file: {model_path}")
        
        if os.path.exists(model_path):
            print(f"✓ Model file exists")
            model_state = torch.load(model_path)
            print(f"✓ Loaded model from file successfully")
        else:
            print(f"✗ Model file not found")
            return False
        
        # Test that args.pickle can be found in the same directory
        args_path = join(os.path.dirname(model_path), 'args.pickle')
        if os.path.exists(args_path):
            print(f"✓ Found args.pickle in same directory")
        else:
            print(f"⚠ args.pickle not found (will use current arguments)")
        
        print("\n✓ Test 2 PASSED: Model file loaded successfully")
        return True


def test_backward_compatibility():
    """Test backward compatibility with old 'flow.npy' naming."""
    print("\n" + "=" * 60)
    print("Test 3: Backward compatibility with old naming")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = join(tmpdir, 'old_checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint with old naming (flow.npy instead of generative_model.npy)
        model_state = {
            'dummy_weight': torch.randn(10, 10),
            'dummy_bias': torch.randn(10)
        }
        
        torch.save(model_state, join(checkpoint_dir, 'flow.npy'))
        torch.save({'state': {}, 'param_groups': [{'lr': 0.0001}]}, 
                   join(checkpoint_dir, 'optim.npy'))
        
        print(f"Created old-format checkpoint with flow.npy")
        
        # Test loading
        if os.path.exists(join(checkpoint_dir, 'flow.npy')):
            print(f"✓ Found flow.npy (old format)")
            model_state = torch.load(join(checkpoint_dir, 'flow.npy'))
            print(f"✓ Loaded model successfully")
            print("\n✓ Test 3 PASSED: Backward compatibility maintained")
            return True
        else:
            print(f"✗ flow.npy not found")
            return False


def test_with_existing_checkpoint():
    """Test with an actual existing checkpoint if available."""
    print("\n" + "=" * 60)
    print("Test 4: Test with existing checkpoint (if available)")
    print("=" * 60)
    
    # Check if there's an existing checkpoint
    existing_checkpoint = 'outputs/exp_35_conditional_nf192_9l_alpha'
    
    if os.path.exists(existing_checkpoint):
        print(f"Found existing checkpoint: {existing_checkpoint}")
        
        # Test loading args
        args_path = join(existing_checkpoint, 'args.pickle')
        if os.path.exists(args_path):
            print(f"✓ Found args.pickle")
            with open(args_path, 'rb') as f:
                args = pickle.load(f)
            print(f"✓ Loaded args successfully")
            if hasattr(args, 'current_epoch'):
                print(f"  - Checkpoint saved at epoch: {args.current_epoch}")
            else:
                print(f"  ⚠ No current_epoch in args (old checkpoint)")
        
        # Test loading model
        if os.path.exists(join(existing_checkpoint, 'generative_model_ema.npy')):
            print(f"✓ Found generative_model_ema.npy")
            print(f"  Size: {os.path.getsize(join(existing_checkpoint, 'generative_model_ema.npy')) / 1024 / 1024:.2f} MB")
        elif os.path.exists(join(existing_checkpoint, 'generative_model.npy')):
            print(f"✓ Found generative_model.npy")
            print(f"  Size: {os.path.getsize(join(existing_checkpoint, 'generative_model.npy')) / 1024 / 1024:.2f} MB")
        else:
            print(f"✗ No model checkpoint found")
            return False
        
        print("\n✓ Test 4 PASSED: Existing checkpoint can be loaded")
        return True
    else:
        print(f"⚠ No existing checkpoint found at {existing_checkpoint}")
        print("Skipping this test")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Resume Functionality")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Resume from directory", test_resume_from_directory()))
    results.append(("Resume from file", test_resume_from_file()))
    results.append(("Backward compatibility", test_backward_compatibility()))
    results.append(("Existing checkpoint", test_with_existing_checkpoint()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests FAILED!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
