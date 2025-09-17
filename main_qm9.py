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

def parse_normalize_factors(values):
    """
    Parse normalize_factors that can be provided in multiple formats:
    - "[1, 2, 3]" (single quoted string)
    - [1, 2, 3] (unquoted, multiple args)
    - 1 2 3 (space-separated numbers)
    """
    if len(values) == 1:
        # Single argument case - could be quoted list or single value
        value = values[0]
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                # Properly formatted list string "[1, 2, 3]"
                try:
                    return eval(value)
                except:
                    pass
            
            # Try to parse as comma-separated values "1,2,3"
            if ',' in value:
                try:
                    # Remove brackets if present
                    value = value.strip('[]')
                    return [float(x.strip()) if '.' in x.strip() else int(x.strip()) for x in value.split(',')]
                except:
                    pass
            
            # Single value "1"
            try:
                return [float(value) if '.' in value else int(value)]
            except:
                pass
    
    # Multiple arguments case - [1, 2, 3] split by shell
    result = []
    for val in values:
        val_str = str(val).strip().rstrip(',').strip('[]')
        try:
            if '.' in val_str:
                result.append(float(val_str))
            else:
                result.append(int(val_str))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Could not parse normalize_factors value: {val}")
    
    return result

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
parser.add_argument('--normalize_factors', nargs='+', type=str, default=['1', '4', '1'],
                    help='normalize factors for [x, categorical, integer]. Can be provided as "[1,4,1]" or 1 4 1')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
parser.add_argument('--export_training_stats', action='store_true',
                    help='Export molecular statistics (molecular_weight, pi_conjugation_ratio, atom_types_encoding, functional_groups_encoding) to CSV files during ASE database training. Only works with --dataset ase_db.')
parser.add_argument('--stats_output_dir', type=str, default='training_stats',
                    help='Directory to save CSV statistics files (default: training_stats)')
args = parser.parse_args()

# Parse normalize_factors from the command line arguments
args.normalize_factors = parse_normalize_factors(args.normalize_factors)

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

dataset_info = get_dataset_info(args.dataset, args.remove_h, getattr(args, 'ase_db_path', None))

# For ASE databases, update normalization factors based on dataset characteristics
if 'ase_db' in args.dataset and hasattr(dataset_info, 'recommended_categorical_norm'):
    recommended_norm = dataset_info.get('recommended_categorical_norm', args.normalize_factors[1])
    original_factors = args.normalize_factors.copy()
    args.normalize_factors[1] = recommended_norm
    print(f"Adjusted categorical normalization factor for ASE database:")
    print(f"  Original factors: {original_factors}")
    print(f"  Updated factors: {args.normalize_factors}")
    print(f"  Reason: Dataset has {dataset_info.get('n_elements', 'unknown')} elements")
elif 'ase_db' in args.dataset:
    # Fallback logic for ASE databases without explicit recommendations
    n_elements = len(dataset_info.get('atom_decoder', []))
    if n_elements > 10:
        original_factors = args.normalize_factors.copy()
        args.normalize_factors[1] = 1.0
        print(f"Applied fallback normalization adjustment for large ASE database:")
        print(f"  Original factors: {original_factors}")
        print(f"  Updated factors: {args.normalize_factors}")
        print(f"  Reason: Dataset has {n_elements} elements")
    elif n_elements > 5:
        original_factors = args.normalize_factors.copy()
        args.normalize_factors[1] = 2.0
        print(f"Applied fallback normalization adjustment for medium ASE database:")
        print(f"  Original factors: {original_factors}")
        print(f"  Updated factors: {args.normalize_factors}")
        print(f"  Reason: Dataset has {n_elements} elements")

# CRITICAL FIX: Check for problematic conditioning combinations that can cause halogen bias
if len(args.conditioning) > 0:
    problematic_features = ['atom_types_encoding', 'functional_groups_encoding']
    found_problematic = [feat for feat in args.conditioning if feat in problematic_features]
    
    if found_problematic:
        # Check if these are properly formatted one-hot encodings after dataset loading
        # This check will be done later after dataloaders are created
        print(f"Note: Using binary features {found_problematic} with ASE database.")
        print(f"These features use improved normalization for stability.")

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

# Automatic learning rate adjustment for ASE databases
if 'ase_db' in args.dataset:
    original_lr = args.lr
    # Use lower learning rate for ASE databases to improve stability
    if args.lr >= 2e-4:  # Only adjust if using default or higher learning rate
        args.lr = 1e-4  # Reduce to more stable learning rate
        print(f"Adjusted learning rate for ASE database stability:")
        print(f"  Original LR: {original_lr}")
        print(f"  Adjusted LR: {args.lr}")
        print(f"  Reason: ASE databases often require lower learning rates for stable training")

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

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

data_dummy = next(iter(dataloaders['train']))

# Export training statistics if requested and using ASE database
if args.export_training_stats:
    if 'ase_db' in args.dataset:
        export_training_statistics(dataloaders, args, args.stats_output_dir)
    else:
        print("⚠️  Warning: --export_training_stats flag ignored.")
        print("   This feature only works with ASE databases (--dataset ase_db).")
        print("   For QM9 datasets, use the existing analysis tools in qm9/analyze.py")


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    
    # Validate binary features for ASE databases after loading
    if args.dataset == 'ase_db':
        binary_features = ['atom_types_encoding', 'functional_groups_encoding']
        used_binary_features = [f for f in binary_features if f in args.conditioning]
        
        if used_binary_features:
            # Check if the binary features are properly formatted as one-hot encodings
            train_data = dataloaders['train'].dataset.data
            
            properly_formatted = []
            problematic = []
            
            for feature in used_binary_features:
                if feature == 'atom_types_encoding':
                    is_onehot = train_data.get('_atom_types_is_onehot', False)
                elif feature == 'functional_groups_encoding':
                    is_onehot = train_data.get('_functional_groups_is_onehot', False)
                else:
                    is_onehot = False
                
                if is_onehot:
                    properly_formatted.append(feature)
                else:
                    problematic.append(feature)
            
            if properly_formatted:
                print(f"✅ Using properly formatted one-hot encodings: {properly_formatted}")
                print("These features are optimized for stable conditional generation.")
            
            if problematic:
                print(f"⚠️  WARNING: Problematic conditioning features detected!")
                print(f"  Features: {problematic}")
                print(f"  These binary features can cause halogen bias during conditional generation.")
                print(f"  Recommendation: Use only scalar features like 'molecular_weight', 'pi_conjugation_ratio'")
        
        if len(args.conditioning) > 3:
            print("Warning: Using many conditioning features may increase training instability.")
            print("Consider starting with 1-2 features and adding more gradually.")
    
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
# Use better initial values for ASE databases based on observed gradient patterns
if args.dataset == 'ase_db':
    # Initialize with multiple values to give better initial statistics
    # Based on the error log, gradients start high but settle around 50-100
    gradnorm_queue.add(50.0)  # Better initial value for ASE databases
    gradnorm_queue.add(30.0)  # Add some variety to initial history
    gradnorm_queue.add(20.0)
    gradnorm_queue.add(40.0)
    gradnorm_queue.add(60.0)  # Now we have 5 values, so adaptive clipping will work
else:
    gradnorm_queue.add(3000)  # Original value for QM9


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def export_training_statistics(dataloaders, args, output_dir='training_stats'):
    """
    Export molecular statistics to CSV files for molecular_weight, pi_conjugation_ratio,
    atom_types_encoding, and functional_groups_encoding.
    
    Parameters:
    -----------
    dataloaders : dict
        Dictionary containing train, valid, test dataloaders
    args : argparse.Namespace
        Command line arguments
    output_dir : str
        Directory to save CSV files
    """
    import os
    import csv
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting training statistics to {output_dir}...")
    
    # Initialize data collectors
    molecular_weights = []
    pi_conjugation_ratios = []
    atom_types_data = []
    functional_groups_data = []
    
    # Check what properties are available in the first batch
    first_batch = next(iter(dataloaders['train']))
    available_properties = list(first_batch.keys())
    print(f"Available properties in dataset: {available_properties}")
    
    # Analyze training data
    print("Analyzing training dataset...")
    train_loader = dataloaders['train']
    
    for batch_idx, data in enumerate(train_loader):
        if batch_idx % 10 == 0:  # Progress indicator
            print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
        
        batch_size = data['positions'].size(0)
        
        for i in range(batch_size):
            # Extract molecular weight if available
            if 'molecular_weight' in data:
                try:
                    mw = data['molecular_weight'][i].item()
                    molecular_weights.append(mw)
                except (AttributeError, IndexError):
                    pass
            
            # Extract pi conjugation ratio if available
            if 'pi_conjugation_ratio' in data:
                try:
                    pi_ratio = data['pi_conjugation_ratio'][i].item()
                    pi_conjugation_ratios.append(pi_ratio)
                except (AttributeError, IndexError):
                    pass
            
            # Extract atom types encoding if available
            if 'atom_types_encoding' in data:
                try:
                    atom_encoding = data['atom_types_encoding'][i].cpu().numpy()
                    atom_types_data.append(atom_encoding)
                except (AttributeError, IndexError):
                    pass
            
            # Extract functional groups encoding if available
            if 'functional_groups_encoding' in data:
                try:
                    fg_encoding = data['functional_groups_encoding'][i].cpu().numpy()
                    functional_groups_data.append(fg_encoding)
                except (AttributeError, IndexError):
                    pass
    
    # Export molecular weight histogram to CSV
    if molecular_weights:
        print(f"Exporting molecular weight histogram ({len(molecular_weights)} samples)...")
        _export_scalar_histogram_csv(
            molecular_weights, 
            os.path.join(output_dir, 'molecular_weight_histogram.csv'),
            'Molecular Weight (u)', 'Count'
        )
    else:
        print("No molecular weight data found - skipping molecular weight export.")
    
    # Export pi conjugation ratio histogram to CSV
    if pi_conjugation_ratios:
        print(f"Exporting pi conjugation ratio histogram ({len(pi_conjugation_ratios)} samples)...")
        _export_scalar_histogram_csv(
            pi_conjugation_ratios,
            os.path.join(output_dir, 'pi_conjugation_ratio_histogram.csv'),
            'Pi Conjugation Ratio', 'Count'
        )
    else:
        print("No pi conjugation ratio data found - skipping pi conjugation ratio export.")
    
    # Export atom types encoding statistics to CSV
    if atom_types_data:
        print(f"Exporting atom types encoding statistics ({len(atom_types_data)} samples)...")
        _export_encoding_statistics_csv(
            atom_types_data,
            os.path.join(output_dir, 'atom_types_encoding_stats.csv'),
            'Atom Type Component', 'Statistics'
        )
    else:
        print("No atom types encoding data found - skipping atom types encoding export.")
    
    # Export functional groups encoding statistics to CSV
    if functional_groups_data:
        print(f"Exporting functional groups encoding statistics ({len(functional_groups_data)} samples)...")
        _export_encoding_statistics_csv(
            functional_groups_data,
            os.path.join(output_dir, 'functional_groups_encoding_stats.csv'),
            'Functional Group Component', 'Statistics'
        )
    else:
        print("No functional groups encoding data found - skipping functional groups encoding export.")
    
    print(f"Statistics export completed. Files saved to {output_dir}/")
    
    # Create a summary report
    summary_file = os.path.join(output_dir, 'export_summary.csv')
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Property', 'Samples Found', 'Files Generated'])
        writer.writerow(['molecular_weight', len(molecular_weights), 
                        2 if molecular_weights else 0])  # histogram + summary
        writer.writerow(['pi_conjugation_ratio', len(pi_conjugation_ratios), 
                        2 if pi_conjugation_ratios else 0])  # histogram + summary
        writer.writerow(['atom_types_encoding', len(atom_types_data), 
                        (len(atom_types_data[0]) + 1) if atom_types_data else 0])  # stats + component histograms
        writer.writerow(['functional_groups_encoding', len(functional_groups_data),
                        (len(functional_groups_data[0]) + 1) if functional_groups_data else 0])  # stats + component histograms


def _export_scalar_histogram_csv(values, filename, value_name, count_name, bins=50):
    """Export histogram of scalar values to CSV."""
    import csv
    import numpy as np
    
    # Handle edge case of empty values
    if not values:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([value_name, count_name])
        return
    
    # Calculate histogram
    hist_counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([value_name, count_name])
        for center, count in zip(bin_centers, hist_counts):
            writer.writerow([f"{center:.4f}", count])
    
    # Also write summary statistics
    summary_filename = filename.replace('.csv', '_summary.csv')
    with open(summary_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Statistic', 'Value'])
        writer.writerow(['Count', len(values)])
        if len(values) > 0:
            writer.writerow(['Mean', f"{np.mean(values):.4f}"])
            writer.writerow(['Std', f"{np.std(values):.4f}"])
            writer.writerow(['Min', f"{np.min(values):.4f}"])
            writer.writerow(['Max', f"{np.max(values):.4f}"])
            writer.writerow(['Q25', f"{np.percentile(values, 25):.4f}"])
            writer.writerow(['Q50 (Median)', f"{np.percentile(values, 50):.4f}"])
            writer.writerow(['Q75', f"{np.percentile(values, 75):.4f}"])


def _export_encoding_statistics_csv(encoding_data, filename, component_name, stats_name):
    """Export statistics for each component of encoding vectors to CSV."""
    import csv
    import numpy as np
    
    if not encoding_data:
        # Create empty file with header
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([component_name, 'Mean', 'Std', 'Min', 'Max', 'Q25', 'Q50', 'Q75', 'Non-zero Count', 'Non-zero %'])
        return
    
    # Convert to numpy array for easier processing
    encoding_array = np.array(encoding_data)  # Shape: (n_molecules, n_components)
    n_components = encoding_array.shape[1]
    
    # Calculate statistics for each component
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([component_name, 'Mean', 'Std', 'Min', 'Max', 'Q25', 'Q50', 'Q75', 'Non-zero Count', 'Non-zero %'])
        
        for i in range(n_components):
            component_values = encoding_array[:, i]
            non_zero_count = np.count_nonzero(component_values)
            non_zero_percent = (non_zero_count / len(component_values)) * 100
            
            writer.writerow([
                f"Component_{i}",
                f"{np.mean(component_values):.6f}",
                f"{np.std(component_values):.6f}",
                f"{np.min(component_values):.6f}",
                f"{np.max(component_values):.6f}",
                f"{np.percentile(component_values, 25):.6f}",
                f"{np.percentile(component_values, 50):.6f}",
                f"{np.percentile(component_values, 75):.6f}",
                non_zero_count,
                f"{non_zero_percent:.2f}%"
            ])
    
    # Export histograms for each component (only for non-zero components)
    for i in range(n_components):
        component_values = encoding_array[:, i]
        non_zero_mask = component_values > 0
        
        if np.any(non_zero_mask):
            # Only create histogram for components that have non-zero values
            non_zero_values = component_values[non_zero_mask]
            component_filename = filename.replace('.csv', f'_component_{i}_histogram.csv')
            unique_values = len(np.unique(non_zero_values))
            bins_to_use = min(20, max(1, unique_values))  # Ensure at least 1 bin
            _export_scalar_histogram_csv(
                non_zero_values.tolist(),
                component_filename,
                f'Component_{i}_Value',
                'Count',
                bins=bins_to_use
            )


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

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

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
