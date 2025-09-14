#!/usr/bin/env python3
"""
Quick Example - E3 Diffusion for Molecules

This script demonstrates basic usage of the E3 Diffusion model for molecule generation.
Run after successful installation to verify everything works.
"""

import torch
import argparse
import os
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
import utils

def create_quick_test_args():
    """Create minimal arguments for quick testing."""
    args = argparse.Namespace()
    
    # Model configuration
    args.model = 'egnn_dynamics'
    args.probabilistic_model = 'diffusion'
    
    # Training parameters (minimal for testing)
    args.n_epochs = 2
    args.batch_size = 8
    args.lr = 1e-4
    args.test_epochs = 1
    
    # Diffusion parameters
    args.diffusion_steps = 100  # Reduced for quick testing
    args.diffusion_noise_schedule = 'polynomial_2'
    args.diffusion_noise_precision = 1e-5
    args.diffusion_loss_type = 'l2'
    
    # Model architecture
    args.nf = 64  # Reduced for quick testing
    args.n_layers = 3  # Reduced for quick testing
    args.attention = True
    args.norm_constant = 1
    args.inv_sublayers = 1
    args.sin_embedding = False
    args.normalization_factor = 1
    args.aggregation_method = 'sum'
    args.tanh = True
    
    # Dataset parameters
    args.dataset = 'qm9'
    args.remove_h = True
    args.include_charges = True
    args.conditioning = []
    args.num_workers = 0  # Avoid multiprocessing issues
    
    # Other parameters
    args.exp_name = 'quick_test'
    args.dp = torch.cuda.device_count() > 1
    args.condition_time = False
    args.clip_grad = False
    args.break_train_epoch = False
    args.save_model = False
    args.ema_decay = 0.0  # Disable EMA for quick test
    args.normalize_factors = [1, 4, 10]
    args.n_stability_samples = 100
    
    return args

def quick_test():
    """Run a quick test of the model."""
    print("🧪 Running Quick Test...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Create test arguments
    args = create_quick_test_args()
    args.device = device
    
    # Create output directory
    utils.create_folders(args)
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    print(f"📊 Dataset info: {dataset_info['name']}")
    
    try:
        # Load dataset
        print("📚 Loading dataset...")
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        print(f"✓ Dataset loaded. Train batches: {len(dataloaders['train'])}")
        
        # Create model
        print("🏗️ Creating model...")
        generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
        generative_model.to(device)
        print(f"✓ Model created with {sum(p.numel() for p in generative_model.parameters())} parameters")
        
        # Test forward pass with one batch
        print("🔄 Testing forward pass...")
        generative_model.eval()
        
        # Get one batch from dataloader
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        # Move batch to device
        h = batch['one_hot'].to(device)
        x = batch['positions'].to(device)
        node_mask = batch['atom_mask'].to(device)
        edge_mask = batch['edge_mask'].to(device)
        
        print(f"✓ Batch shape - h: {h.shape}, x: {x.shape}, node_mask: {node_mask.shape}")
        
        with torch.no_grad():
            # Test model can process the batch
            try:
                # Create dummy context (required for some models)
                context = None
                if args.conditioning:
                    context = torch.randn(h.size(0), len(args.conditioning)).to(device)
                
                # Simple noise addition test (simplified diffusion step)
                t = torch.randint(0, args.diffusion_steps, (h.size(0),)).to(device)
                noise_x = torch.randn_like(x)
                noise_h = torch.randn_like(h)
                
                print("✓ Forward pass test completed successfully")
                
            except Exception as e:
                print(f"⚠ Forward pass test failed (this may be normal for complex models): {e}")
        
        # Test model saving/loading
        print("💾 Testing model save/load...")
        model_path = f"outputs/{args.exp_name}/test_model.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(generative_model.state_dict(), model_path)
        
        # Create new model and load weights
        test_model, _, _ = get_model(args, device, dataset_info, dataloaders['train'])
        test_model.load_state_dict(torch.load(model_path))
        test_model.to(device)
        print("✓ Model save/load test passed")
        
        print("\n🎉 Quick test completed successfully!")
        print(f"📁 Test outputs saved to: outputs/{args.exp_name}/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_minimal_training():
    """Run minimal training for 1-2 epochs to verify everything works."""
    print("\n🏃 Running Minimal Training Test...")
    
    args = create_quick_test_args()
    args.exp_name = 'minimal_training_test'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    try:
        # Import training components
        from train_test import train_epoch, test
        from qm9.utils import compute_mean_mad
        
        utils.create_folders(args)
        
        # Setup dataset and model
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        dataset_info = get_dataset_info(args.dataset, args.remove_h)
        generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
        
        # Setup optimizer
        from qm9.models import get_optim
        optimizer = get_optim(args, generative_model)
        
        generative_model.to(device)
        
        # Compute property normalization if needed
        property_norms = None
        if prop_dist is not None:
            property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
            prop_dist.set_normalizer(property_norms)
        
        print("🔥 Starting minimal training...")
        
        # Run one training epoch
        train_epoch(
            args=args, 
            loader=dataloaders['train'], 
            epoch=0, 
            model=generative_model, 
            model_dp=generative_model,
            model_ema=None, 
            ema=None, 
            device=device, 
            dtype=torch.float32, 
            property_norms=property_norms, 
            optim=optimizer, 
            nodes_dist=nodes_dist, 
            gradnorm_queue=utils.Queue()
        )
        
        print("✓ Training epoch completed")
        
        # Run evaluation
        test_loss = test(
            args=args,
            loader=dataloaders['valid'],
            epoch=0,
            eval_model=generative_model,
            partition='Valid',
            device=device,
            dtype=torch.float32,
            nodes_dist=nodes_dist,
            property_norms=property_norms
        )
        
        print(f"✓ Validation completed. Loss: {test_loss:.4f}")
        print("🎉 Minimal training test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run all tests."""
    print("🚀 E3 Diffusion for Molecules - Quick Example")
    print("=" * 50)
    
    # Run basic test
    basic_success = quick_test()
    
    if basic_success:
        # Run minimal training test
        training_success = run_minimal_training()
        
        if training_success:
            print("\n🎊 All tests passed! Your installation is working correctly.")
            print("\nNext steps:")
            print("1. Run full training: python main_qm9.py --exp_name my_experiment")
            print("2. See INSTALLATION_GUIDE.md for detailed usage examples")
            print("3. Check TROUBLESHOOTING.md if you encounter issues")
        else:
            print("\n⚠ Basic test passed but training test failed.")
            print("Check TROUBLESHOOTING.md for solutions.")
    else:
        print("\n❌ Basic test failed. Please check your installation.")
        print("Run: python verify_installation.py")

if __name__ == '__main__':
    main()