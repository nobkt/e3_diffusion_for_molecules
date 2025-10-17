# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
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

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
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
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset) | ase_db (load from ASE database)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--ase_db_path', type=str, default=None,
                    help='Path to ASE database file (required when using ase_db dataset)')
parser.add_argument('--split_ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    help='Train, validation, test split ratios for ASE database (should sum to 1.0)')
parser.add_argument('--remove_duplicates', type=eval, default=True,
                    help='Remove duplicate molecules from ASE database based on geometry (True/False)')
parser.add_argument('--duplicate_tolerance', type=float, default=1e-6,
                    help='Tolerance for considering two molecules as duplicates (default: 1e-6)')
parser.add_argument('--debug_dataset_csv', type=str, default=None,
                    help='Path to save CSV file with dataset composition analysis (for debugging)')
parser.add_argument('--debug_dataset_xyz', type=str, default=None,
                    help='Directory path to save XYZ files for all molecules (for debugging)')
parser.add_argument('--ase_to_eV', type=str, default='{}',
                    help='JSON string of unit conversion factors for ASE data (e.g., {"energy": 27.2114})')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv | molecular_weight | pi_conjugation_ratio | atom_types_encoding | functional_groups_encoding' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=100,
                    help="Normalize the sum aggregation of EGNN. Higher values (e.g., 100) provide better numerical stability.")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
parser.add_argument('--export_conditions_csv', type=str, default=None,
                    help='Export generation conditions (molecular_weight, pi_conjugation_ratio, '
                         'atom_types_encoding, functional_groups_encoding) to CSV files and exit. '
                         'Specify output directory path.')
args = parser.parse_args()

# For ASE databases, atomic charges are not included in the database
# so we should default to include_charges=False
if 'ase_db' in args.dataset and args.include_charges:
    print("Warning: For ASE databases, atomic charges are not included in the database.")
    print("Setting include_charges=False automatically.")
    args.include_charges = False

# Parse unit conversion for ASE data
import json
try:
    args.ase_to_eV = json.loads(args.ase_to_eV)
except json.JSONDecodeError:
    print(f"Warning: Could not parse ase_to_eV argument: {args.ase_to_eV}")
    args.ase_to_eV = {}

# Validate ASE database arguments
if 'ase_db' in args.dataset and args.ase_db_path is None:
    raise ValueError("--ase_db_path must be specified when using ase_db dataset")

# Validate split ratios
if abs(sum(args.split_ratios) - 1.0) > 1e-6:
    raise ValueError(f"Split ratios must sum to 1.0, got {sum(args.split_ratios)}")

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    import os
    
    # Store command-line arguments that should override saved args
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    # Determine where to load args.pickle from
    if os.path.isdir(args.resume):
        args_path = join(args.resume, 'args.pickle')
    else:
        # If resume is a file, look for args.pickle in the same directory
        args_path = join(os.path.dirname(args.resume), 'args.pickle')
    
    # Load saved arguments
    if os.path.exists(args_path):
        print(f"Loading arguments from {args_path}")
        with open(args_path, 'rb') as f:
            args = pickle.load(f)
    else:
        print(f"Warning: No args.pickle found at {args_path}. Using current arguments.")

    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name
    args.wandb_usr = wandb_usr

    # Load saved dataset_info if available to ensure consistent model architecture
    if os.path.isdir(args.resume):
        dataset_info_path = join(args.resume, 'dataset_info.pickle')
    else:
        dataset_info_path = join(os.path.dirname(args.resume), 'dataset_info.pickle')
    
    saved_dataset_info = None
    if os.path.exists(dataset_info_path):
        print(f"Loading dataset_info from {dataset_info_path}")
        with open(dataset_info_path, 'rb') as f:
            saved_dataset_info = pickle.load(f)
        # Store the saved atom encoder/decoder to restore later
        saved_atom_encoder = saved_dataset_info['atom_encoder']
        saved_atom_decoder = saved_dataset_info['atom_decoder']
        print(f"Saved dataset has {len(saved_atom_decoder)} atom types: {saved_atom_decoder}")
    else:
        print(f"Warning: No dataset_info.pickle found at {dataset_info_path}. Will use current dataset configuration.")
        saved_atom_encoder = None
        saved_atom_decoder = None

    # Handle start_epoch: Use the saved current_epoch if start_epoch was not explicitly set
    # (start_epoch default is 0, so we check if it was explicitly provided)
    if start_epoch == 0 and hasattr(args, 'current_epoch'):
        # Use the epoch from the checkpoint
        args.start_epoch = args.current_epoch
        print(f"Resuming from epoch {args.start_epoch} (from checkpoint)")
    else:
        # Use the explicitly provided start_epoch
        args.start_epoch = start_epoch
        if start_epoch > 0:
            print(f"Resuming from epoch {args.start_epoch} (from command line)")

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(f"Resume configuration: exp_name={args.exp_name}, start_epoch={args.start_epoch}")
    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

# Restore saved dataset_info if resuming training
if args.resume is not None and 'saved_atom_encoder' in dir() and saved_atom_encoder is not None:
    print(f"Restoring saved atom types to dataset_info to ensure model architecture matches checkpoint")
    dataset_info['atom_encoder'] = saved_atom_encoder
    dataset_info['atom_decoder'] = saved_atom_decoder
    print(f"Restored dataset_info with {len(saved_atom_decoder)} atom types: {saved_atom_decoder}")

# Check if CSV export mode is enabled
if args.export_conditions_csv is not None:
    print("\n" + "="*60)
    print("CSV Export Mode Enabled")
    print("="*60)
    
    # Get datasets from dataloaders
    datasets = {}
    for split_name, dataloader in dataloaders.items():
        datasets[split_name] = dataloader.dataset
    
    # Export generation conditions to CSV
    from qm9.dataset import export_generation_conditions_to_csv
    export_generation_conditions_to_csv(
        datasets=datasets,
        output_dir=args.export_conditions_csv,
        dataset_info=dataset_info
    )
    
    print("\nCSV export completed. Exiting program.")
    exit(0)

data_dummy = next(iter(dataloaders['train']))


# When resuming, preserve context_node_nf from saved args to ensure model architecture matches checkpoint
if args.resume is not None and hasattr(args, 'context_node_nf'):
    # Use the saved context_node_nf from the checkpoint
    context_node_nf = args.context_node_nf
    print(f'Resuming training: using saved context_node_nf = {context_node_nf}')
    
    # Still compute property_norms for conditioning
    if len(args.conditioning) > 0:
        print(f'Conditioning on {args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    else:
        property_norms = None
    
    # Ensure args.context_node_nf is preserved (redundant but explicit for safety)
    args.context_node_nf = context_node_nf
else:
    # Normal training: calculate context_node_nf from data
    if len(args.conditioning) > 0:
        print(f'Conditioning on {args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
    else:
        context_node_nf = 0
        property_norms = None

    args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(100)  # Start with reasonable value instead of extremely large one


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    if args.resume is not None:
        import os
        # Support both directory path and file path for resume argument
        if os.path.isdir(args.resume):
            # If resume is a directory, look for model files in it
            resume_dir = args.resume
            
            # Try to load EMA model first (usually better), then regular model
            if os.path.exists(join(resume_dir, 'generative_model_ema.npy')):
                model_path = join(resume_dir, 'generative_model_ema.npy')
                print(f"Loading EMA model from {model_path}")
            elif os.path.exists(join(resume_dir, 'generative_model.npy')):
                model_path = join(resume_dir, 'generative_model.npy')
                print(f"Loading model from {model_path}")
            elif os.path.exists(join(resume_dir, 'flow.npy')):
                # Backward compatibility with old naming
                model_path = join(resume_dir, 'flow.npy')
                print(f"Loading model from {model_path} (old format)")
            else:
                raise FileNotFoundError(
                    f"No model checkpoint found in {resume_dir}. "
                    f"Expected generative_model_ema.npy, generative_model.npy, or flow.npy"
                )
            
            optim_path = join(resume_dir, 'optim.npy')
        else:
            # If resume is a file path, use it directly for the model
            model_path = args.resume
            # Try to find optimizer in the same directory
            resume_dir = os.path.dirname(args.resume)
            optim_path = join(resume_dir, 'optim.npy')
            print(f"Loading model from {model_path}")
        
        # Load model state
        flow_state_dict = torch.load(model_path)
        model.load_state_dict(flow_state_dict)
        
        # Load optimizer state if available
        if os.path.exists(optim_path):
            print(f"Loading optimizer state from {optim_path}")
            optim_state_dict = torch.load(optim_path)
            optim.load_state_dict(optim_state_dict)
        else:
            print(f"Warning: Optimizer state not found at {optim_path}. Starting with fresh optimizer.")

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)
                    # Save dataset_info for resume compatibility
                    with open('outputs/%s/dataset_info.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(dataset_info, f)

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
                    # Save dataset_info for resume compatibility
                    with open('outputs/%s/dataset_info_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(dataset_info, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
