"""
Main training script for property-conditioned crystal generation.

This script extends main_crystal.py to support property conditioning,
implementing Phase 3 of the property-conditioned molecular crystal generation system.

Key Features:
- Property-conditioned training using CrystalDatasetWithProperties
- ExtendedCombinedConditioning with property module
- Property statistics saved with checkpoints
- Resume training with property conditioning
- Property-specific metrics logging

Design Principles:
- No fallback heuristics (ごまかしのためのfallbackは絶対にしない)
- E(3) equivariance maintained
- Modular and backward compatible
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
from train_test_crystal import train_epoch_crystal, test_crystal, analyze_and_save_crystal

# Crystal-specific imports
from crystal.data.crystal_loader import CrystalDatasetWithProperties, collate_crystal_batch
from crystal.data.molecule_loader import MoleculeDataset
from crystal.data.molecule_crystal_mapper import MoleculeCrystalMapper
from crystal.models.molecular_encoder import MolecularEncoder
from crystal.models.crystal_dynamics import CrystalDynamics
from crystal.conditioning import (
    MolecularConditioning,
    SpaceGroupEmbedding,
    DensityConditioning,
    PropertyConditioning,
    ExtendedCombinedConditioning
)
from crystal.evaluation import CrystalMetrics, StructureValidator
from crystal.utils import CIFWriter


parser = argparse.ArgumentParser(description='E3 Diffusion for Property-Conditioned Crystal Generation')

# Experiment settings
parser.add_argument('--exp_name', type=str, default='crystal_with_properties',
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
parser.add_argument('--condition_on_property', type=eval, default=True,
                    help='Condition on physical properties')
parser.add_argument('--molecular_encoder_path', type=str, default=None,
                    help='Path to pretrained molecular encoder')
parser.add_argument('--conditioning_dim', type=int, default=256,
                    help='Dimension of conditioning vector')

# Property conditioning settings (NEW)
parser.add_argument('--property_names', nargs='+', type=str, default=None,
                    help='List of property names to condition on (e.g., bandgap melting_point)')
parser.add_argument('--property_hidden_dim', type=int, default=512,
                    help='Hidden dimension for property MLP')
parser.add_argument('--property_n_layers', type=int, default=3,
                    help='Number of layers in property MLP')

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
                    help='Path to checkpoint to resume from')

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


def load_crystal_data_with_properties(args, device):
    """Load crystal and molecule datasets with property support."""
    print("Loading crystal data with properties...")
    
    # Validate property conditioning arguments
    if args.condition_on_property:
        if args.property_names is None or len(args.property_names) == 0:
            raise ValueError(
                "Property conditioning enabled but no property names provided. "
                "Use --property_names to specify properties (e.g., --property_names bandgap melting_point)"
            )
        print(f"Property conditioning enabled for: {args.property_names}")
    
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
    
    # Load crystal dataset with properties
    print(f"Loading crystal dataset from {args.crystal_db_path}")
    
    if args.condition_on_property:
        # Use CrystalDatasetWithProperties for property-conditioned training
        # First get all indices from database
        from ase.db import connect
        db = connect(args.crystal_db_path)
        all_indices = [row.id for row in db.select()]
        db._close()
        
        crystal_dataset = CrystalDatasetWithProperties(
            db_path=args.crystal_db_path,
            indices=all_indices,
            property_names=args.property_names,
            molecule_dataset=molecule_dataset,
            molecule_crystal_mapper=mapper,
            use_fractional_coords=args.use_fractional_coords,
            max_atoms=args.max_atoms,
            remove_h=args.remove_h
        )
        
        print(f"Property statistics:")
        for i, prop_name in enumerate(args.property_names):
            mean = crystal_dataset.property_mean[i].item()
            std = crystal_dataset.property_std[i].item()
            print(f"  {prop_name}: mean={mean:.4f}, std={std:.4f}")
    else:
        # Use standard CrystalDataset
        from crystal.data.crystal_loader import CrystalDataset
        from ase.db import connect
        db = connect(args.crystal_db_path)
        all_indices = [row.id for row in db.select()]
        db._close()
        
        crystal_dataset = CrystalDataset(
            db_path=args.crystal_db_path,
            indices=all_indices,
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


def setup_models_with_properties(args, device, dataset_info, crystal_dataset, molecule_dataset=None):
    """Initialize models and conditioning modules with property support."""
    print("Setting up models with property conditioning...")
    
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
    
    # Initialize conditioning modules
    mol_cond = None
    sg_embed = None
    dens_cond = None
    prop_cond = None
    
    # Molecular conditioning
    if args.condition_on_molecule and mol_encoder is not None:
        mol_cond = MolecularConditioning(
            molecular_feature_dim=args.nf,
            conditioning_dim=args.conditioning_dim,
            use_geometry=True
        ).to(device)
    
    # Space group embedding
    if args.condition_on_space_group:
        sg_embed = SpaceGroupEmbedding(
            num_space_groups=230,
            embedding_dim=args.conditioning_dim
        ).to(device)
    
    # Density conditioning
    if args.condition_on_density:
        dens_cond = DensityConditioning(
            conditioning_dim=args.conditioning_dim
        ).to(device)
    
    # Property conditioning (NEW)
    if args.condition_on_property:
        prop_cond = PropertyConditioning(
            property_names=args.property_names,
            conditioning_dim=args.conditioning_dim,
            hidden_dim=args.property_hidden_dim,
            n_layers=args.property_n_layers
        ).to(device)
        
        # Set normalization parameters from dataset statistics
        prop_cond.set_normalization_params(
            crystal_dataset.property_mean,
            crystal_dataset.property_std
        )
        print("Property conditioning initialized with dataset statistics")
    
    # Combined conditioning using ExtendedCombinedConditioning
    combined_conditioning = ExtendedCombinedConditioning(
        molecular_conditioning=mol_cond,
        space_group_embedding=sg_embed,
        density_conditioning=dens_cond,
        property_conditioning=prop_cond,
        conditioning_dim=args.conditioning_dim
    ).to(device)
    
    # Store individual modules for checkpointing
    conditioning_modules = {
        'combined': combined_conditioning
    }
    if mol_cond is not None:
        conditioning_modules['molecular'] = mol_cond
    if sg_embed is not None:
        conditioning_modules['space_group'] = sg_embed
    if dens_cond is not None:
        conditioning_modules['density'] = dens_cond
    if prop_cond is not None:
        conditioning_modules['property'] = prop_cond
    
    context_node_nf = args.conditioning_dim
    
    # Create dynamics model
    print(f"Creating CrystalDynamics model with context_node_nf={context_node_nf}")
    dynamics = CrystalDynamics(
        in_node_nf=dataset_info['input_nf'],
        hidden_nf=args.nf,
        n_layers=args.n_layers,
        context_node_nf=context_node_nf,
        device=device,
        tanh=args.tanh,
        attention=args.attention,
        norm_constant=args.norm_constant,
        sin_embedding=args.sin_embedding,
        learn_lattice=args.learn_lattice,
        lattice_hidden_dim=args.lattice_hidden_dim,
        lattice_num_layers=args.lattice_num_layers,
        use_fractional_coords=args.use_fractional_coords,
        periodic_cutoff=args.periodic_cutoff
    ).to(device)
    
    # Wrap in diffusion model
    vdm = en_diffusion.EnVariationalDiffusion(
        dynamics=dynamics,
        in_node_nf=dataset_info['input_nf'],
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=None  # Will compute from data
    )
    
    model = vdm.to(device)
    
    # Setup optimizer
    optim = get_optim(args, model)
    
    # Setup EMA
    if args.ema_decay > 0:
        ema = flow_utils.EMA(args.ema_decay)
        ema_model = copy.deepcopy(model)
    else:
        ema = None
        ema_model = model
    
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, mol_encoder, conditioning_modules, optim, ema, ema_model


def save_checkpoint_with_properties(args, epoch, model, optim, mol_encoder, conditioning_modules,
                                   crystal_dataset, best_val_loss, checkpoint_path):
    """Save checkpoint including property statistics."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'mol_encoder_state_dict': mol_encoder.state_dict() if mol_encoder else None,
        'conditioning_modules': {k: v.state_dict() for k, v in conditioning_modules.items()},
        'best_val_loss': best_val_loss,
        'args': args
    }
    
    # Add property statistics if using property conditioning
    if args.condition_on_property:
        checkpoint['property_names'] = args.property_names
        checkpoint['property_mean'] = crystal_dataset.property_mean
        checkpoint['property_std'] = crystal_dataset.property_std
        print(f"Saved property statistics for: {args.property_names}")
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint_with_properties(args, checkpoint_path, model, optim, mol_encoder, 
                                   conditioning_modules, device):
    """Load checkpoint including property statistics."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if mol_encoder and checkpoint.get('mol_encoder_state_dict'):
        mol_encoder.load_state_dict(checkpoint['mol_encoder_state_dict'])
    
    if checkpoint.get('conditioning_modules'):
        for k, v in checkpoint['conditioning_modules'].items():
            if k in conditioning_modules:
                conditioning_modules[k].load_state_dict(v)
    
    # Restore property statistics if present
    if args.condition_on_property and 'property_names' in checkpoint:
        loaded_prop_names = checkpoint['property_names']
        if loaded_prop_names != args.property_names:
            raise ValueError(
                f"Property names mismatch! Checkpoint has {loaded_prop_names}, "
                f"but args specify {args.property_names}"
            )
        
        # Restore normalization parameters to property conditioning module
        prop_cond = conditioning_modules.get('property')
        if prop_cond:
            prop_cond.set_normalization_params(
                checkpoint['property_mean'],
                checkpoint['property_std']
            )
        print(f"Restored property statistics for: {loaded_prop_names}")
    
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    start_epoch = checkpoint['epoch'] + 1
    
    return start_epoch, best_val_loss


def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    print("\n" + "="*70)
    print("Property-Conditioned Crystal Generation - Training Script")
    print("="*70)
    print(f"Using device: {device}")
    print(f"Property conditioning: {args.condition_on_property}")
    if args.condition_on_property:
        print(f"Properties: {args.property_names}")
    print(f"Arguments: {args}")
    
    # Setup W&B
    setup_wandb(args)
    
    # Load data with properties
    dataloaders, crystal_dataset, molecule_dataset = load_crystal_data_with_properties(args, device)
    
    # Get dataset info
    dataset_info = get_dataset_info('qm9' if not args.remove_h else 'qm9_without_h', args.remove_h)
    
    # Setup models with property conditioning
    model, mol_encoder, conditioning_modules, optim, ema, ema_model = setup_models_with_properties(
        args, device, dataset_info, crystal_dataset, molecule_dataset
    )
    
    # Resume from checkpoint if specified
    best_val_loss = float('inf')
    if args.resume:
        args.start_epoch, best_val_loss = load_checkpoint_with_properties(
            args, args.resume, model, optim, mol_encoder, conditioning_modules, device
        )
        print(f"Resumed from epoch {args.start_epoch}, best val loss: {best_val_loss:.4f}")
    
    # DataParallel wrapper
    if args.dp and torch.cuda.device_count() > 1:
        model_dp = torch.nn.DataParallel(model)
    else:
        model_dp = model
    
    # Setup evaluation tools
    crystal_metrics = CrystalMetrics(dataset_info)
    structure_validator = StructureValidator()
    cif_writer = CIFWriter(dataset_info) if args.save_cif else None
    
    # Setup gradient norm queue
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed
    
    # Nodes distribution (for sampling)
    nodes_dist = None  # Placeholder for now
    
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
    dtype = torch.float32
    
    print("\n" + "="*70)
    print("Starting property-conditioned crystal training loop")
    print("="*70 + "\n")
    
    for epoch in range(args.start_epoch, args.n_epochs):
        print(f"\nEpoch {epoch+1}/{args.n_epochs}")
        start_time = time.time()
        
        # Training
        train_epoch_crystal(
            args=args,
            loader=dataloaders['train'],
            epoch=epoch,
            model=model,
            model_dp=model_dp,
            model_ema=ema_model,
            ema=ema,
            device=device,
            dtype=dtype,
            mol_encoder=mol_encoder,
            conditioning_modules=conditioning_modules,
            optim=optim,
            nodes_dist=nodes_dist,
            gradnorm_queue=gradnorm_queue,
            dataset_info=dataset_info
        )
        
        # Validation
        if epoch % args.test_epochs == 0:
            print("\nValidating...")
            val_loss = test_crystal(
                args=args,
                loader=dataloaders['valid'],
                epoch=epoch,
                eval_model=ema_model,
                device=device,
                dtype=dtype,
                mol_encoder=mol_encoder,
                conditioning_modules=conditioning_modules,
                nodes_dist=nodes_dist,
                partition='Val'
            )
            
            print(f"Validation NLL: {val_loss:.2f}")
            wandb.log({"Val NLL": val_loss}, commit=True)
            
            # Analyze and save structures
            if args.validate_structures:
                print("\nAnalyzing generated structures...")
                metrics = analyze_and_save_crystal(
                    epoch=epoch,
                    model_sample=ema_model,
                    nodes_dist=nodes_dist,
                    args=args,
                    device=device,
                    dataset_info=dataset_info,
                    crystal_metrics=crystal_metrics,
                    structure_validator=structure_validator,
                    cif_writer=cif_writer,
                    mol_encoder=mol_encoder,
                    conditioning_modules=conditioning_modules,
                    n_samples=args.n_stability_samples,
                    batch_size=10
                )
            
            # Save checkpoint with property statistics
            if args.save_model:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch{epoch:04d}.pt'
                save_checkpoint_with_properties(
                    args, epoch, model, optim, mol_encoder, conditioning_modules,
                    crystal_dataset, best_val_loss, checkpoint_path
                )
                
                # Save best model separately
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = checkpoint_dir / 'best_model.pt'
                    print(f"New best model! Saving to {best_model_path}")
                    save_checkpoint_with_properties(
                        args, epoch, model, optim, mol_encoder, conditioning_modules,
                        crystal_dataset, best_val_loss, best_model_path
                    )
        
        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.2f}s")
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation NLL: {best_val_loss:.2f}")
    print("="*70)
    wandb.finish()


if __name__ == '__main__':
    main()
