import argparse
from os.path import join
import torch
import pickle
from qm9.models import get_model
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from qm9.sampling import sample_chain, sample, sample_sweep_conditional, sample_exact_conditional
import qm9.visualizer as vis
import ast
import re


def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def parse_property_values(property_values_str):
    """
    Parse property values string into a dictionary of property names and values.
    
    Format: 'molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O],functional_groups_encoding=[[CX3](=O)[OX2H1],[NX3;H2,H1;!$(NC=O)]]'
    
    Returns:
        dict: Dictionary with property names as keys and parsed values
    """
    if not property_values_str:
        return {}
    
    parsed_values = {}
    
    # Split by commas, but be careful with nested brackets
    # Use a more sophisticated parsing approach
    current_prop = ""
    bracket_depth = 0
    paren_depth = 0
    
    for char in property_values_str + ',':  # Add comma at end to process last item
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == ',' and bracket_depth == 0 and paren_depth == 0:
            # Process current property
            if current_prop.strip():
                prop_name, prop_value = current_prop.split('=', 1)
                prop_name = prop_name.strip()
                prop_value = prop_value.strip()
                
                # Parse the value based on its format
                parsed_values[prop_name] = parse_single_property_value(prop_value)
            current_prop = ""
            continue
        
        current_prop += char
    
    return parsed_values


def parse_single_property_value(value_str):
    """
    Parse a single property value string into appropriate Python type.
    
    Args:
        value_str: String representation of the value
        
    Returns:
        Parsed value (float, list, etc.)
    """
    value_str = value_str.strip()
    
    # Try to parse as float first
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Try to parse as list
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            # Handle atom types like [C,H,N,O]
            if ',' in value_str and not '(' in value_str:
                # Simple list of atoms
                content = value_str[1:-1]  # Remove brackets
                items = [item.strip() for item in content.split(',')]
                return items
            else:
                # More complex list, try literal_eval
                return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # If parsing fails, return as string
            return value_str
    
    # Return as string if no other parsing worked
    return value_str


def create_exact_context(property_values, args_gen, property_norms, n_frames, n_nodes, device):
    """
    Create context tensor with exact property values that matches the model's expected dimensions.
    
    Args:
        property_values: Dictionary of property names and exact values
        args_gen: Generator arguments with conditioning information
        property_norms: Property normalization parameters
        n_frames: Number of frames to generate
        n_nodes: Number of nodes per molecule
        device: Device to create tensors on
        
    Returns:
        torch.Tensor: Context tensor with exact conditions repeated for all frames
    """
    import numpy as np
    
    # Check if model expects any context at all
    if not hasattr(args_gen, 'context_node_nf') or args_gen.context_node_nf == 0:
        return None
    
    expected_context_features = args_gen.context_node_nf
    
    # Create a context tensor with the exact expected dimensions
    # Initialize with zeros (normalized mean for most properties)
    context = torch.zeros(n_frames, expected_context_features, dtype=torch.float32, device=device)
    
    # Fill in the context features based on available exact values
    # We need to match the order and dimensions used during training
    feature_idx = 0
    
    for key in args_gen.conditioning:
        if feature_idx >= expected_context_features:
            break
            
        if key in property_values and key in property_norms:
            exact_value = property_values[key]
            
            if isinstance(exact_value, (int, float)):
                # Scalar property - normalize and set
                mean = property_norms[key]['mean']
                mad = property_norms[key]['mad']
                
                # Handle tensor means/mads
                if hasattr(mean, 'item'):
                    mean = mean.item()
                if hasattr(mad, 'item'):
                    mad = mad.item()
                    
                normalized_value = (exact_value - mean) / mad
                context[:, feature_idx] = normalized_value
                feature_idx += 1
                
            elif isinstance(exact_value, list) and key == 'atom_types_encoding':
                # For atom types, we might use a simpler encoding that fits the expected dimensions
                # If we have multiple atom types, we might encode them as a single feature
                # or use only the first few features
                from configs.datasets_config import get_dataset_info
                dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
                atom_encoder = dataset_info.get('atom_encoder', {})
                
                # Determine how many features this property should use
                mean = property_norms[key]['mean']
                if hasattr(mean, 'numel'):
                    n_features = mean.numel()
                elif hasattr(mean, 'shape'):
                    n_features = mean.shape[0] if len(mean.shape) == 1 else 1
                else:
                    n_features = 1
                
                # Don't exceed the expected context size
                n_features = min(n_features, expected_context_features - feature_idx)
                
                if n_features > 0:
                    # Create encoding that fits the expected dimensions
                    encoding = torch.zeros(n_features, dtype=torch.float32)
                    
                    # Simple encoding: set first len(exact_value) features to 1
                    for i, atom_symbol in enumerate(exact_value[:n_features]):
                        if atom_symbol in atom_encoder:
                            encoding[i] = 1.0
                    
                    # Apply normalization if available
                    mean_tensor = property_norms[key]['mean']
                    mad_tensor = property_norms[key]['mad']
                    
                    if hasattr(mean_tensor, 'shape') and len(mean_tensor.shape) > 0:
                        mean_vals = mean_tensor[:n_features] if len(mean_tensor) >= n_features else mean_tensor
                        mad_vals = mad_tensor[:n_features] if len(mad_tensor) >= n_features else mad_tensor
                        encoding = (encoding - mean_vals) / mad_vals
                    
                    context[:, feature_idx:feature_idx + n_features] = encoding.unsqueeze(0).repeat(n_frames, 1)
                    feature_idx += n_features
                    
            else:
                # For other properties, use default normalized values (usually zero)
                mean = property_norms[key]['mean']
                if hasattr(mean, 'numel'):
                    n_features = min(mean.numel(), expected_context_features - feature_idx)
                elif hasattr(mean, 'shape') and len(mean.shape) > 0:
                    n_features = min(mean.shape[0], expected_context_features - feature_idx)
                else:
                    n_features = min(1, expected_context_features - feature_idx)
                
                # Keep default zeros (which represent normalized means)
                feature_idx += n_features
        else:
            # Property not in exact values or property_norms - use defaults
            # Determine expected feature count for this property
            if key in property_norms:
                mean = property_norms[key]['mean']
                if hasattr(mean, 'numel'):
                    n_features = min(mean.numel(), expected_context_features - feature_idx)
                elif hasattr(mean, 'shape') and len(mean.shape) > 0:
                    n_features = min(mean.shape[0], expected_context_features - feature_idx)
                else:
                    n_features = min(1, expected_context_features - feature_idx)
            else:
                n_features = min(1, expected_context_features - feature_idx)
            
            # Keep default zeros
            feature_idx += n_features
    
    print(f"Created context with shape {context.shape}, expected {expected_context_features} features per frame")
    return context


def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    # Support both qm9_second_half and ase_db datasets
    assert args_gen.dataset in ['qm9_second_half', 'ase_db'], f"Unsupported dataset: {args_gen.dataset}"

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_generator(dir_path, dataloaders, device, args_gen, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model, nodes_dist, prop_dist = get_model(args_gen, device, dataset_info, dataloaders['train'])
    fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


class DiffusionDataloader:
    def __init__(self, args_gen, model, nodes_dist, prop_dist, device, unkown_labels=False,
                 batch_size=1, iterations=200, property_key=None):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0
        # Use the specified property, or fall back to the first one
        self.property_key = property_key if property_key is not None else (self.prop_dist.properties[0] if self.prop_dist else None)

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)

        node_mask = node_mask.squeeze(2)
        context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        prop_key = self.property_key
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        else:
            context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        data = {
            'positions': x.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            'one_hot': one_hot.detach(),
            prop_key: context.detach()
        }
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, _ = get_generator(args.generators_path, dataloaders,
                                                    args.device, args_gen, property_norms)

    # Verify that the requested property was part of the conditioning during training
    if args.property not in args_gen.conditioning:
        raise ValueError(f"Property '{args.property}' was not used for conditioning during training. "
                        f"Available conditioning properties: {args_gen.conditioning}")

    # Create a dataloader with the generator

    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    if args.task == 'edm':
        diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist,
                                                   args.device, batch_size=args.batch_size, iterations=args.iterations, property_key=args.property)
        print("EDM: We evaluate the classifier on our generated samples")
        loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
        print("Loss classifier on Generated samples: %.4f" % loss)
    elif args.task == 'qm9_second_half':
        print(f"{args_gen.dataset}: We evaluate the classifier on the training dataset")
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print(f"Loss classifier on {args_gen.dataset}: %.4f" % loss)
    elif args.task == 'naive':
        print("Naive: We evaluate the classifier on QM9")
        length = dataloaders['train'].dataset.data[args.property].size(0)
        idxs = torch.randperm(length)
        dataloaders['train'].dataset.data[args.property] = dataloaders['train'].dataset.data[args.property][idxs]
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on naive: %.4f" % loss)
    #elif args.task == 'numnodes':
    #    print("Numnodes: We evaluate the numnodes classifier on EDM samples")
    #    diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist, device,
    #                                               batch_size=args.batch_size, iterations=args.iterations)
    #    loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
    #    print("Loss numnodes classifier on EDM generated samples: %.4f" % loss)


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0, exact_context=None):
    """
    Sample and save conditional molecules.
    
    Args:
        args: Arguments with conditioning information
        device: Device to run on  
        model: Trained generative model
        prop_dist: Property distribution
        dataset_info: Dataset information
        epoch: Epoch number for naming
        id_from: Starting ID for naming
        exact_context: If provided, use exact conditions instead of sweep
    """
    if exact_context is not None:
        # Use exact conditional sampling
        one_hot, charges, x, node_mask = sample_exact_conditional(args, device, model, dataset_info, prop_dist, exact_context)
    else:
        # Use sweep conditional sampling (original behavior)
        one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/analysis/run%s/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    vis.visualize_chain("outputs/%s/analysis/run%s/" % (args.exp_name, epoch), dataset_info,
                        wandb=None, mode='conditional', spheres_3d=True)

    return one_hot, charges, x


def main_qualitative(args):
    args_gen = get_args_gen(args.generators_path)
    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path,
                                                               dataloaders, args.device, args_gen,
                                                               property_norms)

    # Parse exact property values if provided
    exact_context = None
    if args.use_exact_conditions and args.property_values:
        property_values = parse_property_values(args.property_values)
        print(f"Using exact conditions: {property_values}")
        
        # Create exact context tensor
        n_frames = 100  # Default number of molecules to generate
        n_nodes = 19    # Default number of nodes (can be made configurable)
        exact_context = create_exact_context(property_values, args_gen, property_norms, 
                                            n_frames, n_nodes, args.device)
        
        print(f"Created exact context tensor with shape: {exact_context.shape if exact_context is not None else None}")

    for i in range(args.n_sweeps):
        print("Sampling sweep %d/%d" % (i+1, args.n_sweeps))
        if exact_context is not None:
            print("  Using exact conditional sampling")
        else:
            print("  Using sweep conditional sampling")
        save_and_sample_conditional(args_gen, args.device, model, prop_dist, dataset_info, 
                                   epoch=i, id_from=0, exact_context=exact_context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_alpha')
    parser.add_argument('--generators_path', type=str, default='outputs/exp_cond_alpha_pretrained')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--property', type=str, default='alpha',
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv', 'molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding'")
    parser.add_argument('--property_values', type=str, default=None,
                        help="Specify exact property values as key=value pairs, e.g., 'molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]'")
    parser.add_argument('--use_exact_conditions', action='store_true',
                        help='Use exact property values instead of property sweeps for generation')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=20,
                        help='break point or not')
    parser.add_argument('--task', type=str, default='qualitative',
                        help='naive, edm, qm9_second_half, qualitative')
    parser.add_argument('--n_sweeps', type=int, default=10,
                        help='number of sweeps for the qualitative conditional experiment')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.task == 'qualitative':
        main_qualitative(args)
    else:
        main_quantitative(args)
