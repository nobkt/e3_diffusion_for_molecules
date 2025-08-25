#!/usr/bin/env python3
"""
Main script for training/testing the diffusion model with ASE database support.
This extends the original main_qm9.py to support ASE databases.
"""

import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save


def create_ase_parser():
    """Create argument parser with ASE database support."""
    parser = argparse.ArgumentParser(description='E3Diffusion with ASE Database Support')
    
    # Original arguments
    parser.add_argument('--exp_name', type=str, default='ase_debug')
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | '
                             'kernel_dynamics | egnn_dynamics |gnn_dynamics')
    parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                        help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='learned, cosine')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='vlb, l2')

    parser.add_argument('--n_epochs', type=int, default=5)  # Reduced for testing
    parser.add_argument('--batch_size', type=int, default=32)  # Reduced for testing
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--brute_force', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--actnorm', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--break_train_epoch', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--dp', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--condition_time', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--clip_grad', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--trace', type=str, default='hutch',
                        help='hutch | exact')

    # EGNN args
    parser.add_argument('--n_layers', type=int, default=9,
                        help='number of layers')
    parser.add_argument('--inv_sublayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nf', type=int, default=256,
                        help='number of layers')
    parser.add_argument('--tanh', type=eval, default=True,
                        help='use tanh in the coord_mlp')
    parser.add_argument('--attention', type=eval, default=True,
                        help='use attention in the EGNN')
    parser.add_argument('--norm_diff', type=eval, default=False,
                        help='normalize the coordinates difference')
    parser.add_argument('--coord_agg', type=str, default='mean',
                        help='mean | sum')

    # Data args
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='qm9 | geom | ase_<name>')
    parser.add_argument('--datadir', type=str, default='qm9/temp',
                        help='Data directory')
    parser.add_argument('--filter_n_atoms', type=int, default=None,
                        help='Filter molecules by number of atoms')
    parser.add_argument('--include_charges', type=eval, default=True,
                        help='include atom charge or not')
    parser.add_argument('--remove_h', type=eval, default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')

    # ASE-specific arguments
    parser.add_argument('--ase_db_path', type=str, default=None,
                        help='Path to ASE database file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio for ASE databases')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Validation set ratio for ASE databases')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio for ASE databases')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for ASE database splits')

    # Conditional arguments
    parser.add_argument('--conditioning', nargs='+', default=[],
                        help='Properties to condition on')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch to start from')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Amount of EMA decay, 0 means off. A reasonable value'
                             ' is 0.999.')
    parser.add_argument('--augment_noise', type=float, default=0)
    parser.add_argument('--n_report_steps', type=int, default=1)

    parser.add_argument('--wandb_usr', type=str)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save_model', type=eval, default=True,
                        help='save model')

    parser.add_argument('--generate_epochs', type=int, default=1,
                        help='save model')
    parser.add_argument('--num_epochs', type=int, default=3000,
                        help='save model')
    parser.add_argument('--test_epochs', type=int, default=1)
    parser.add_argument('--data_augmentation', type=eval, default=False, help="Data augmentation")
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1], help="normalize factors")

    return parser


def setup_ase_dataset(args):
    """Setup ASE dataset configuration."""
    if 'ase' in args.dataset or args.ase_db_path:
        # Update dataset info for ASE
        if args.ase_db_path:
            from qm9.ase_database import ASEDatabaseReader
            ase_reader = ASEDatabaseReader(args.ase_db_path)
            dataset_info = ase_reader.get_dataset_info(args.dataset, with_h=not args.remove_h)
        else:
            dataset_info = get_dataset_info(args.dataset, args.remove_h)
    else:
        dataset_info = get_dataset_info(args.dataset, args.remove_h)
    
    return dataset_info


def main_ase():
    parser = create_ase_parser()
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    
    # Setup dataset
    dataset_info = setup_ase_dataset(args)
    
    print(f"Dataset: {args.dataset}")
    print(f"Atom types: {dataset_info['atom_decoder']}")
    print(f"Max nodes: {dataset_info['max_n_nodes']}")
    
    # Get dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    
    # Create model
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, datadir=args.datadir)
    
    # Setup optimizer
    optim = get_optim(args, model)
    print(f"Model: {model}")
    
    # Setup for training
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value so that first few iterations are not skipped.

    # Get property normalizations
    if len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, dataset_info)
    else:
        property_norms = None

    # Optionally resume from checkpoint
    if args.resume is not None:
        # Load checkpoint logic here if needed
        pass

    # Create output directory
    args.main_path = args.datadir + '/' + args.exp_name
    args.models_path = args.main_path + '/models/'
    args.summaries_path = args.main_path + '/summaries/'
    
    if not utils.check_mask_correct(vars(args), device):
        raise utils.FoundNaNException("Something wrong with the mask")

    # Initialize wandb if user provided
    if args.wandb_usr:
        wandb.init(project="e3_diffusion_molecules", 
                   name=args.exp_name,
                   config=args)

    best_nll_val = 1e8
    best_nll_test = 1e8
    
    print("Starting training...")
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        
        # Training
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch,
                   model=model, model_dp=model, model_ema=None, ema=None, 
                   device=device, dtype=dtype, property_norms=property_norms,
                   optim=optim, nodes_dist=nodes_dist, gradnorm_queue=gradnorm_queue,
                   dataset_info=dataset_info, prop_dist=prop_dist)
        
        print(f"Epoch {epoch} took {time.time() - start_epoch:.1f} seconds")
        
        # Validation/Testing
        if epoch % args.test_epochs == 0:
            if len(args.conditioning) > 0:
                val_nll = test(args=args, loader=dataloaders['valid'], epoch=epoch,
                              eval_model=model, partition='Valid', device=device, dtype=dtype,
                              nodes_dist=nodes_dist, property_norms=property_norms)
                test_nll = test(args=args, loader=dataloaders['test'], epoch=epoch,
                               eval_model=model, partition='Test', device=device, dtype=dtype,
                               nodes_dist=nodes_dist, property_norms=property_norms)
                
                if val_nll < best_nll_val:
                    best_nll_val = val_nll
                    best_nll_test = test_nll
                    if args.save_model:
                        args.current_epoch = epoch + 1
                        utils.save_model(optim, 'outputs/%s' % args.exp_name, epoch, model, args)
                
                print(f"Val loss: {val_nll:.4f} | Test loss: {test_nll:.4f} | "
                      f"Best val: {best_nll_val:.4f} | Best test: {best_nll_test:.4f}")
        
        # Generation/Analysis
        if epoch % args.generate_epochs == 0:
            print(f"Analyzing epoch {epoch}...")
            
            analyze_and_save(args=args, device=device, generative_model=model,
                           nodes_dist=nodes_dist, prop_dist=prop_dist,
                           dataset_info=dataset_info, epoch=epoch,
                           batch_size=args.batch_size)

    print("Training completed!")
    
    # Final evaluation
    print("\nFinal evaluation...")
    analyze_and_save(args=args, device=device, generative_model=model,
                    nodes_dist=nodes_dist, prop_dist=prop_dist,
                    dataset_info=dataset_info, epoch=args.n_epochs,
                    batch_size=args.batch_size)


if __name__ == "__main__":
    main_ase()