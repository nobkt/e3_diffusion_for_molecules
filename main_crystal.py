"""
Main training script for crystal generation using E(3) Equivariant Diffusion Model.

This script integrates Phase 1-5 components:
- Data loading with periodic boundary conditions
- Molecular EGNN feature extraction
- Crystal dynamics model with lattice learning
- Conditioning on molecular features, space groups, and density
- Evaluation and CIF output

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- Strict validation throughout
- Theoretically sound crystal generation
"""

# Rdkit import should be first
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import argparse
import copy
import pickle
import time
import torch
import wandb
from os.path import join
from pathlib import Path

import utils
from configs.datasets_config import get_dataset_info
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
from qm9.models import get_optim
from qm9.utils import compute_mean_mad
from train_test import train_epoch, test

# Crystal-specific imports
from crystal.data.crystal_loader import CrystalDataset, collate_crystal_batch
from crystal.data.molecule_loader import MoleculeDataset
from crystal.data.molecule_crystal_mapper import MoleculeCrystalMapper
from crystal.models.molecular_encoder import MolecularEncoder
from crystal.models.crystal_dynamics import CrystalDynamics
from crystal.conditioning import (
    MolecularConditioning,
    SpaceGroupEmbedding,
    DensityConditioning,
    CombinedConditioning
)
from crystal.evaluation import CrystalMetrics, StructureValidator
from crystal.utils import CIFWriter


parser = argparse.ArgumentParser(description='E3 Diffusion for Crystal Generation')

# Experiment settings
parser.add_argument('--exp_name', type=str, default='crystal_debug',
                    help='Experiment name')
parser.add_argument('--model', type=str, default='crystal_dynamics',
                    help='Model type: crystal_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='Probabilistic model: diffusion')

# Diffusion settings
parser.add_argument('--diffusion_steps', type=int, default=500,
                    help='Number of diffusion steps')
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='Noise schedule: learned, cosine, polynomial_2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    help='Noise precision')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='Loss type: vlb, l2')

# Training settings
parser.add_argument('--n_epochs', type=int, default=200,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size (smaller for crystals)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Starting epoch (for resuming)')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='Break after first batch (debugging)')
parser.add_argument('--test_epochs', type=int, default=10,
                    help='Test every N epochs')
parser.add_argument('--n_stability_samples', type=int, default=100,
                    help='Number of samples for stability evaluation')

# Model architecture
parser.add_argument('--n_layers', type=int, default=6,
                    help='Number of EGNN layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='Number of invariant sublayers')
parser.add_argument('--nf', type=int, default=128,
                    help='Hidden feature dimension')
parser.add_argument('--tanh', type=eval, default=True,
                    help='Use tanh in coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='Use attention in EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='Normalization constant for distances')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='Use sin embedding for distances')

# Crystal-specific settings
parser.add_argument('--learn_lattice', type=eval, default=True,
                    help='Learn lattice parameters')
parser.add_argument('--lattice_hidden_dim', type=int, default=128,
                    help='Hidden dimension for lattice diffusion')
parser.add_argument('--lattice_num_layers', type=int, default=3,
                    help='Number of layers for lattice diffusion')
parser.add_argument('--periodic_cutoff', type=float, default=10.0,
                    help='Cutoff for periodic neighbor list (Angstroms)')
parser.add_argument('--use_fractional_coords', type=eval, default=True,
                    help='Use fractional coordinates')

# Data settings
parser.add_argument('--crystal_db_path', type=str, required=True,
                    help='Path to crystal ASE database')
parser.add_argument('--molecule_db_path', type=str, default=None,
                    help='Path to molecule ASE database (for molecular features)')
parser.add_argument('--split_ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    help='Train, validation, test split ratios')
parser.add_argument('--max_atoms', type=int, default=None,
                    help='Maximum number of atoms per crystal')
parser.add_argument('--remove_h', action='store_true',
                    help='Remove hydrogen atoms')

# Conditioning settings
parser.add_argument('--condition_on_molecule', type=eval, default=True,
                    help='Condition on molecular features')
parser.add_argument('--condition_on_space_group', type=eval, default=False,
                    help='Condition on space group')
parser.add_argument('--condition_on_density', type=eval, default=False,
                    help='Condition on density')
parser.add_argument('--molecular_encoder_path', type=str, default=None,
                    help='Path to pretrained molecular encoder')
parser.add_argument('--conditioning_dim', type=int, default=256,
                    help='Dimension of conditioning vector')

# Optimization settings
parser.add_argument('--dp', type=eval, default=True,
                    help='Use DataParallel')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='Condition on time step')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='Clip gradients')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='EMA decay rate (0 to disable)')
parser.add_argument('--trace', type=str, default='hutch',
                    help='Trace estimator: hutch | exact')
parser.add_argument('--ode_regularization', type=float, default=1e-3,
                    help='ODE regularization weight')

# Evaluation and output
parser.add_argument('--save_cif', type=eval, default=True,
                    help='Save generated crystals as CIF files')
parser.add_argument('--cif_output_dir', type=str, default=None,
                    help='Directory for CIF output (default: outputs/{exp_name}/cif/)')
parser.add_argument('--validate_structures', type=eval, default=True,
                    help='Validate generated structures')
parser.add_argument('--n_report_steps', type=int, default=10,
                    help='Report every N steps')

# W&B and logging
parser.add_argument('--wandb_usr', type=str, default=None,
                    help='W&B username')
parser.add_argument('--no_wandb', action='store_true',
                    help='Disable W&B logging')
parser.add_argument('--online', type=bool, default=True,
                    help='W&B online mode')

# Resume training
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint directory to resume from')

# Other
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('--save_model', type=eval, default=True,
                    help='Save model checkpoints')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='Generate samples every N epochs')

args = parser.parse_args()


def setup_wandb(args):
    """Initialize W&B logging."""
    args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
    
    if args.resume is not None:
        exp_name = args.exp_name + '_resume'
    else:
        exp_name = args.exp_name
    
    utils.create_folders(args)
    
    if args.no_wandb:
        mode = 'disabled'
    else:
        mode = 'online' if args.online else 'offline'
    
    kwargs = {
        'entity': args.wandb_usr,
        'name': exp_name,
        'project': 'e3_diffusion_crystal',
        'config': args,
        'settings': wandb.Settings(_disable_stats=True),
        'reinit': True,
        'mode': mode
    }
    wandb.init(**kwargs)
    wandb.save('*.txt')


def load_crystal_data(args, device):
    """Load crystal and molecule datasets."""
    print("Loading crystal data...")
    
    # Determine if we should use molecular features
    use_molecular_features = args.condition_on_molecule and args.molecule_db_path is not None
    
    # Load molecule dataset if needed
    if use_molecular_features:
        print(f"Loading molecule dataset from {args.molecule_db_path}")
        molecule_dataset = MoleculeDataset(
            db_path=args.molecule_db_path,
            remove_h=args.remove_h
        )
        
        # Build molecule-crystal mapper
        mapper = MoleculeCrystalMapper()
        mapper.build_from_databases(args.molecule_db_path, args.crystal_db_path)
    else:
        molecule_dataset = None
        mapper = None
    
    # Load crystal dataset
    print(f"Loading crystal dataset from {args.crystal_db_path}")
    crystal_dataset = CrystalDataset(
        db_path=args.crystal_db_path,
        molecule_dataset=molecule_dataset,
        molecule_crystal_mapper=mapper,
        use_fractional_coords=args.use_fractional_coords,
        max_atoms=args.max_atoms,
        remove_h=args.remove_h
    )
    
    # Split dataset
    n_total = len(crystal_dataset)
    n_train = int(n_total * args.split_ratios[0])
    n_val = int(n_total * args.split_ratios[1])
    n_test = n_total - n_train - n_val
    
    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n_total))
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(crystal_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(crystal_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(crystal_dataset, test_indices)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_crystal_batch,
        num_workers=0  # Start with 0 for debugging
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystal_batch,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystal_batch,
        num_workers=0
    )
    
    dataloaders = {
        'train': train_loader,
        'valid': val_loader,
        'test': test_loader
    }
    
    print(f"Dataset split: train={n_train}, val={n_val}, test={n_test}")
    
    return dataloaders, crystal_dataset, molecule_dataset


def setup_models(args, device, dataset_info, molecule_dataset=None):
    """Initialize models and conditioning modules."""
    print("Setting up models...")
    
    # Molecular encoder (if using molecular conditioning)
    if args.condition_on_molecule and molecule_dataset is not None:
        mol_encoder = MolecularEncoder(
            in_node_nf=dataset_info['input_nf'],
            hidden_nf=args.nf,
            n_layers=args.n_layers,
            global_feature_dim=args.nf
        )
        
        # Load pretrained if provided
        if args.molecular_encoder_path is not None:
            print(f"Loading pretrained molecular encoder from {args.molecular_encoder_path}")
            state = torch.load(args.molecular_encoder_path, map_location=device)
            mol_encoder.load_state_dict(state)
        
        mol_encoder = mol_encoder.to(device)
    else:
        mol_encoder = None
    
    # Conditioning modules
    conditioning_modules = {}
    context_node_nf = 0
    
    if args.condition_on_molecule and mol_encoder is not None:
        mol_cond = MolecularConditioning(
            molecular_feature_dim=args.nf,
            conditioning_dim=args.conditioning_dim,
            use_geometry=True
        ).to(device)
        conditioning_modules['molecular'] = mol_cond
        context_node_nf = args.conditioning_dim
    else:
        # If using space group or density without molecular features,
        # we need CombinedConditioning to handle the fusion
        if args.condition_on_space_group or args.condition_on_density:
            # Create a dummy molecular conditioning for combined fusion
            # Note: This is a limitation - CombinedConditioning requires molecular features
            raise NotImplementedError(
                "Space group and density conditioning currently require molecular "
                "conditioning to be enabled. Set --condition_on_molecule True."
            )
    
    if args.condition_on_space_group:
        sg_emb = SpaceGroupEmbedding(
            embedding_dim=64,
            num_space_groups=230
        ).to(device)
        conditioning_modules['space_group'] = sg_emb
    
    if args.condition_on_density:
        dens_cond = DensityConditioning(
            embedding_dim=64,
            density_min=0.5,
            density_max=5.0
        ).to(device)
        conditioning_modules['density'] = dens_cond
    
    # Crystal dynamics model
    model = CrystalDynamics(
        in_node_nf=dataset_info['input_nf'],
        hidden_nf=args.nf,
        n_layers=args.n_layers,
        context_node_nf=context_node_nf if context_node_nf > 0 else 0,
        learn_lattice=args.learn_lattice,
        lattice_hidden_dim=args.lattice_hidden_dim,
        lattice_num_layers=args.lattice_num_layers,
        attention=args.attention,
        tanh=args.tanh,
        norm_constant=args.norm_constant
    )
    
    model = model.to(device)
    
    # Optimizer
    optim = get_optim(args, model)
    
    # EMA
    if args.ema_decay > 0:
        ema = flow_utils.EMA(args.ema_decay)
        ema_model = copy.deepcopy(model)
    else:
        ema = None
        ema_model = model
    
    return model, mol_encoder, conditioning_modules, optim, ema, ema_model


def main():
    """Main training loop."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # Setup W&B
    setup_wandb(args)
    
    # Load data
    dataloaders, crystal_dataset, molecule_dataset = load_crystal_data(args, device)
    
    # Get dataset info
    dataset_info = get_dataset_info('qm9' if not args.remove_h else 'qm9_without_h', args.remove_h)
    
    # Setup models
    model, mol_encoder, conditioning_modules, optim, ema, ema_model = setup_models(
        args, device, dataset_info, molecule_dataset
    )
    
    # Setup evaluation tools
    crystal_metrics = CrystalMetrics(dataset_info)
    structure_validator = StructureValidator()
    cif_writer = CIFWriter(dataset_info) if args.save_cif else None
    
    # Create output directories
    output_dir = Path('outputs') / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    if args.save_cif:
        if args.cif_output_dir is None:
            cif_output_dir = output_dir / 'cif'
        else:
            cif_output_dir = Path(args.cif_output_dir)
        cif_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "="*50)
    print("IMPORTANT NOTE: This is a training script template.")
    print("The actual training loop requires adaptation of train_epoch() and test()")
    print("functions from train_test.py to handle crystal data with periodic boundaries.")
    print("="*50 + "\n")
    
    for epoch in range(args.start_epoch, args.n_epochs):
        print(f"\nEpoch {epoch+1}/{args.n_epochs}")
        start_time = time.time()
        
        # Training
        # TODO: Implement crystal-specific training loop
        # This requires adapting train_epoch() to handle:
        # - Periodic boundary conditions
        # - Cell parameter learning
        # - Molecular feature conditioning
        # - Multiple conditioning types
        print("Training epoch... (template - needs implementation)")
        
        # Placeholder: Would call adapted train_epoch here
        # train_loss = train_epoch_crystal(
        #     args, dataloaders['train'], epoch, model, device,
        #     mol_encoder, conditioning_modules, optim, ...
        # )
        
        # Validation
        if epoch % args.test_epochs == 0:
            print("Validating... (template - needs implementation)")
            
            # Placeholder: Would call adapted test here
            # val_loss = test_crystal(
            #     args, dataloaders['valid'], epoch, ema_model, device, ...
            # )
            
            # Save checkpoint
            if args.save_model:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch{epoch:04d}.pt'
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'mol_encoder_state_dict': mol_encoder.state_dict() if mol_encoder else None,
                    'conditioning_modules': {k: v.state_dict() for k, v in conditioning_modules.items()},
                    'best_val_loss': best_val_loss,
                    'args': args
                }, checkpoint_path)
                
                # Save best model separately
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     best_model_path = checkpoint_dir / 'best_model.pt'
                #     torch.save(model.state_dict(), best_model_path)
        
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.2f}s")
    
    print("\n" + "="*50)
    print("Training loop template completed.")
    print("To use this script for actual training, implement:")
    print("1. train_epoch_crystal() - crystal-specific training")
    print("2. test_crystal() - crystal-specific validation")
    print("3. Adapt conditioning preparation for crystal data")
    print("="*50)
    wandb.finish()


if __name__ == "__main__":
    main()
