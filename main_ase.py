#!/usr/bin/env python3
"""
Training script for EDM with ASE databases.
This script provides the same functionality as main_qm9.py and main_geom_drugs.py
but for custom ASE database datasets.
"""

import argparse
import torch
import wandb
import copy
import time
import pickle
import os
from os.path import join

# Import E3 Diffusion components
from qm9.dataset import retrieve_dataloaders
from qm9.models import get_model, get_optim
from qm9.utils import prepare_context, compute_mean_mad
from equivariant_diffusion import en_diffusion
from equivariant_diffusion import utils as flow_utils
from train_test import train_epoch, test, analyze_and_save
import utils


def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='EDM Training with ASE Database')
    
    # Dataset parameters
    parser.add_argument('--ase_db_path', type=str, required=True,
                        help='Path to ASE database file')
    parser.add_argument('--exp_name', type=str, default='edm_ase',
                        help='Experiment name')
    parser.add_argument('--max_atoms', type=int, default=100,
                        help='Maximum number of atoms per molecule')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                        help='Model type: egnn_dynamics | simple_dynamics | gnn_dynamics')
    parser.add_argument('--nf', type=int, default=256,
                        help='Number of features')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--attention', type=bool, default=True,
                        help='Use attention in EGNN')
    parser.add_argument('--tanh', type=bool, default=True,
                        help='Use tanh activation')
    
    # Diffusion parameters
    parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                        help='Type of probabilistic model')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='Number of diffusion steps')
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='Noise schedule: polynomial_2 | cosine | learned')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                        help='Noise precision')
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='Loss type: l2 | vlb')
    
    # Training options
    parser.add_argument('--test_epochs', type=int, default=10,
                        help='Test every N epochs')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--normalize_factors', type=float, nargs=3, default=[1, 4, 10],
                        help='Normalization factors for [coordinates, features, charges]')
    parser.add_argument('--n_stability_samples', type=int, default=500,
                        help='Number of samples for stability analysis')
    
    # Advanced options
    parser.add_argument('--include_charges', action='store_true', default=True,
                        help='Include atomic charges in the model')
    parser.add_argument('--conditioning', type=str, default='',
                        help='Property conditioning (empty for no conditioning)')
    parser.add_argument('--context_node_nf', type=int, default=0,
                        help='Context node features')
    parser.add_argument('--norm_constant', type=float, default=1.0,
                        help='Normalization constant')
    parser.add_argument('--inv_sublayers', type=int, default=2,
                        help='Number of invariant sublayers')
    parser.add_argument('--sin_embedding', action='store_true', default=False,
                        help='Use sinusoidal embedding')
    parser.add_argument('--normalization_factor', type=float, default=1.0,
                        help='Normalization factor for dynamics')
    parser.add_argument('--aggregation_method', type=str, default='sum',
                        help='Aggregation method for EGNN')
    parser.add_argument('--augment_noise', type=float, default=0.0,
                        help='Noise augmentation factor')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch number')
    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='Use data augmentation')
    parser.add_argument('--trace', type=str, default='hutch',
                        help='Trace method')
    parser.add_argument('--ode_regularization', type=float, default=1e-3,
                        help='ODE regularization')
    parser.add_argument('--dequantization', type=str, default='deterministic',
                        help='Dequantization method')
    parser.add_argument('--n_report_steps', type=int, default=1,
                        help='Number of report steps')
    parser.add_argument('--visualize_every_batch', type=int, default=int(1e8),
                        help='Visualize every N batches')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--dp', action='store_true',
                        help='Use DataParallel for multi-GPU training')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    parser.add_argument('--clip_grad', action='store_true',
                        help='Clip gradients')
    parser.add_argument('--break_train_epoch', action='store_true',
                        help='Break training epochs early for debugging')
    parser.add_argument('--condition_time', action='store_true', default=True,
                        help='Use time conditioning in diffusion')
    parser.add_argument('--subtract_thermo', action='store_true', default=False,
                        help='Subtract thermodynamic quantities')
    
    # Wandb options
    parser.add_argument('--wandb_project', type=str, default='edm_ase',
                        help='Wandb project name')
    parser.add_argument('--wandb_usr', type=str, default=None,
                        help='Wandb username/entity')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--online', type=bool, default=True,
                        help='True = wandb online -- False = wandb offline')
    
    return parser.parse_args()


def setup_device_and_dtype(args):
    """Setup device and data type."""
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    dtype = torch.float32
    return device, dtype


def setup_dataset(args, device):
    """Setup dataset and dataloaders."""
    print(f"Loading ASE database: {args.ase_db_path}")
    
    # Configure dataset loading for ASE
    class Config:
        dataset = 'ase_custom'  # Will trigger ASE dataset loading
        ase_db_path = args.ase_db_path
        batch_size = args.batch_size
        num_workers = args.num_workers
        max_atoms = args.max_atoms
        include_charges = True
        datadir = './data'
        remove_h = False
        filter_n_atoms = None
        sequential = False
        # device will be passed to retrieve_dataloaders
    
    # Set device on config after class creation
    config = Config()
    config.device = device
    
    # Set device on config after class creation
    config = Config()
    config.device = device
    
    # Load dataset
    dataloaders, charge_scale = retrieve_dataloaders(config)
    dataset_info = dataloaders['train'].dataset_info
    
    print(f"Dataset loaded successfully:")
    print(f"  Atom types: {dataset_info['atom_decoder']}")
    print(f"  Max atoms: {dataset_info['max_n_nodes']}")
    print(f"  Training samples: {len(dataloaders['train'].dataset)}")
    print(f"  Validation samples: {len(dataloaders['valid'].dataset)}")
    print(f"  Test samples: {len(dataloaders['test'].dataset)}")
    
    return dataloaders, charge_scale, dataset_info


def setup_model_and_optim(args, device, dataset_info, charge_scale):
    """Setup model and optimizer."""
    print("Initializing model...")
    
    # Get model (returns model, nodes_dist, prop_dist)
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, charge_scale)
    
    # Get optimizer
    optim = get_optim(args, model)
    
    print(f"Model initialized:")
    print(f"  Type: {args.model}")
    print(f"  Features: {args.nf}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize gradnorm queue (this might not be needed for ASE)
    gradnorm_queue = None
    
    return model, optim, gradnorm_queue, nodes_dist, prop_dist


def setup_distributions(dataset_info):
    """Setup node and property distributions."""
    # Node distribution
    nodes_dist = flow_utils.DistributionNodes(dataset_info['n_nodes'])
    
    # Property distribution (currently not used for ASE datasets)
    prop_dist = None
    
    return nodes_dist, prop_dist


def setup_ema(args, model, device):
    """Setup exponential moving average."""
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)
        
        # Setup DataParallel for EMA if needed
        if args.dp and torch.cuda.device_count() > 1 and args.cuda:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model
    
    return model_ema, model_ema_dp, ema


def setup_output_dir(args):
    """Setup output directory."""
    output_dir = f'outputs/{args.exp_name}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir


def setup_wandb(args):
    """Setup Weights & Biases logging."""
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    
    # Import utils to get wandb username
    import utils
    wandb_usr = utils.get_wandb_username(args.wandb_usr)
    
    wandb.init(
        entity=wandb_usr,
        project=args.wandb_project,
        name=args.exp_name,
        config=vars(args),
        mode=mode,
        settings=wandb.Settings(_disable_stats=True),
        reinit=True
    )
    
    if args.no_wandb:
        print("Wandb logging disabled")
    else:
        print("Wandb logging enabled")


def resume_from_checkpoint(args, model, optim):
    """Resume training from checkpoint if specified."""
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        
        # Load model state
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
        model.load_state_dict(flow_state_dict)
        
        # Load optimizer state
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        optim.load_state_dict(optim_state_dict)
        
        print("Checkpoint loaded successfully")


def save_checkpoint(args, epoch, model, model_ema, optim, is_best=False):
    """Save model checkpoint."""
    if args.save_model:
        output_dir = f'outputs/{args.exp_name}'
        
        # Save current model
        utils.save_model(optim, f'{output_dir}/optim.npy')
        utils.save_model(model, f'{output_dir}/generative_model.npy')
        
        if args.ema_decay > 0:
            utils.save_model(model_ema, f'{output_dir}/generative_model_ema.npy')
        
        # Save args
        with open(f'{output_dir}/args.pickle', 'wb') as f:
            pickle.dump(args, f)
        
        # Save epoch-specific checkpoint
        if is_best:
            utils.save_model(optim, f'{output_dir}/optim_{epoch}.npy')
            utils.save_model(model, f'{output_dir}/generative_model_{epoch}.npy')
            if args.ema_decay > 0:
                utils.save_model(model_ema, f'{output_dir}/generative_model_ema_{epoch}.npy')
            with open(f'{output_dir}/args_{epoch}.pickle', 'wb') as f:
                pickle.dump(args, f)


def training_loop(args, dataloaders, model, model_dp, model_ema, model_ema_dp, ema, 
                  optim, device, dtype, dataset_info, nodes_dist, prop_dist, 
                  property_norms, gradnorm_queue):
    """Main training loop."""
    print("Starting training...")
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    
    for epoch in range(args.n_epochs):
        start_time = time.time()
        
        # Train epoch
        train_epoch(
            args=args,
            loader=dataloaders['train'],
            epoch=epoch,
            model=model,
            model_dp=model_dp,
            model_ema=model_ema,
            ema=ema,
            device=device,
            dtype=dtype,
            property_norms=property_norms,
            nodes_dist=nodes_dist,
            dataset_info=dataset_info,
            gradnorm_queue=gradnorm_queue,
            optim=optim,
            prop_dist=prop_dist
        )
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:4d} took {epoch_time:.1f}s")
        
        # Evaluation
        if epoch % args.test_epochs == 0:
            # Log diffusion model information
            if isinstance(model, en_diffusion.EnVariationalDiffusion) and not args.no_wandb:
                wandb.log(model.log_info(), commit=True)
            
            # Analyze stability and save samples
            if not args.break_train_epoch:
                analyze_and_save(
                    args=args,
                    epoch=epoch,
                    model_sample=model_ema,
                    nodes_dist=nodes_dist,
                    dataset_info=dataset_info,
                    device=device,
                    prop_dist=prop_dist,
                    n_samples=args.n_stability_samples
                )
            
            # Validation
            nll_val = test(
                args=args,
                loader=dataloaders['valid'],
                epoch=epoch,
                eval_model=model_ema_dp,
                partition='Val',
                device=device,
                dtype=dtype,
                nodes_dist=nodes_dist,
                property_norms=property_norms
            )
            
            # Test
            nll_test = test(
                args=args,
                loader=dataloaders['test'],
                epoch=epoch,
                eval_model=model_ema_dp,
                partition='Test',
                device=device,
                dtype=dtype,
                nodes_dist=nodes_dist,
                property_norms=property_norms
            )
            
            # Check if this is the best model
            is_best = nll_val < best_nll_val
            if is_best:
                best_nll_val = nll_val
                best_nll_test = nll_test
            
            # Save checkpoint
            save_checkpoint(args, epoch, model, model_ema, optim, is_best)
            
            # Print results
            print(f'Val loss: {nll_val:.4f} | Test loss: {nll_test:.4f}')
            print(f'Best val: {best_nll_val:.4f} | Best test: {best_nll_test:.4f}')
            
            # Log to wandb
            if not args.no_wandb:
                wandb.log({
                    "epoch": epoch,
                    "val_loss": nll_val,
                    "test_loss": nll_test,
                    "best_val_loss": best_nll_val,
                    "best_test_loss": best_nll_test,
                    "epoch_time": epoch_time
                })
    
    print("Training completed!")
    return best_nll_val, best_nll_test


def main():
    """Main function."""
    # Setup
    args = setup_args()
    device, dtype = setup_device_and_dtype(args)
    
    # Setup output directory
    output_dir = setup_output_dir(args)
    
    # Setup wandb
    setup_wandb(args)
    
    # Load dataset
    dataloaders, charge_scale, dataset_info = setup_dataset(args, device)
    
    # Setup model and optimizer
    model, optim, gradnorm_queue, nodes_dist, prop_dist = setup_model_and_optim(args, device, dataset_info, charge_scale)
    
    # Setup distributions (already done in setup_model_and_optim)
    # nodes_dist, prop_dist = setup_distributions(dataset_info)
    
    # Setup context and normalization
    # For ASE datasets, we don't have the same properties as QM9, so we'll use dummy normalization
    property_norms = {
        'coordinates': {'mean': 0.0, 'mad': 1.0},
        'charges': {'mean': 0.0, 'mad': 1.0},
        'features': {'mean': 0.0, 'mad': 1.0}
    }
    
    # Resume from checkpoint if specified
    resume_from_checkpoint(args, model, optim)
    
    # Setup DataParallel if needed
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Using {torch.cuda.device_count()} GPUs for training')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model
    
    # Setup EMA
    model_ema, model_ema_dp, ema = setup_ema(args, model, device)
    
    # Training loop
    best_val, best_test = training_loop(
        args, dataloaders, model, model_dp, model_ema, model_ema_dp, ema,
        optim, device, dtype, dataset_info, nodes_dist, prop_dist,
        property_norms, gradnorm_queue
    )
    
    print(f"Training finished!")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Best test loss: {best_test:.4f}")
    print(f"Model saved to: {output_dir}")
    
    # Finish wandb
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()