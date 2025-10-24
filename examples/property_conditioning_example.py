"""
Example: Property-Conditioned Crystal Generation

This example demonstrates how to use the property conditioning modules
to generate molecular crystals with specific target properties.

Based on: doc/gen_molecular_crystal/README.md
"""

import torch
from crystal.conditioning import (
    PropertyConditioning,
    MolecularConditioning,
    ExtendedCombinedConditioning,
    SpaceGroupEmbedding,
    DensityConditioning,
)
from crystal.data.property_normalizer import PropertyNormalizer


def example_property_conditioning():
    """Example 1: Basic property conditioning."""
    print("=" * 70)
    print("Example 1: Basic Property Conditioning")
    print("=" * 70)
    
    # Define properties we want to condition on
    property_names = ['bandgap', 'melting_point', 'dielectric_constant']
    
    # Create property conditioning module
    prop_cond = PropertyConditioning(
        property_names=property_names,
        conditioning_dim=256,
        hidden_dim=512,
        n_layers=3
    )
    
    # Set normalization parameters (from training data statistics)
    mean = torch.tensor([2.5, 180.0, 3.0])
    std = torch.tensor([1.2, 50.0, 0.8])
    prop_cond.set_normalization_params(mean, std)
    
    # Example target properties
    target_properties = torch.tensor([
        [2.5, 180.0, 3.0],  # Target: bandgap=2.5, melting_point=180, dielectric=3.0
        [3.0, 200.0, 3.5],  # Target: bandgap=3.0, melting_point=200, dielectric=3.5
    ])
    
    # Get conditioning vectors
    conditioning_vectors = prop_cond(target_properties)
    
    print(f"Property names: {property_names}")
    print(f"Target properties shape: {target_properties.shape}")
    print(f"Conditioning vectors shape: {conditioning_vectors.shape}")
    print(f"Target properties[0]: {target_properties[0].tolist()}")
    print(f"Conditioning vector[0] (first 5 dims): {conditioning_vectors[0][:5].tolist()}")
    print()


def example_property_normalizer():
    """Example 2: Property normalizer usage."""
    print("=" * 70)
    print("Example 2: Property Normalizer")
    print("=" * 70)
    
    # Create normalizer with statistics
    mean = torch.tensor([2.5, 180.0, 3.0])
    std = torch.tensor([1.2, 50.0, 0.8])
    property_names = ['bandgap', 'melting_point', 'dielectric_constant']
    
    normalizer = PropertyNormalizer(mean, std, property_names)
    
    # Original properties
    properties = torch.tensor([
        [2.5, 180.0, 3.0],
        [3.7, 230.0, 3.8],
    ])
    
    # Normalize
    normalized = normalizer.normalize(properties)
    
    # Denormalize back
    recovered = normalizer.denormalize(normalized)
    
    print(f"Original properties:\n{properties}")
    print(f"\nNormalized properties:\n{normalized}")
    print(f"\nRecovered properties:\n{recovered}")
    print(f"\nReconstruction error: {(properties - recovered).abs().max().item():.6f}")
    print()


def example_combined_conditioning():
    """Example 3: Combined conditioning with all condition types."""
    print("=" * 70)
    print("Example 3: Extended Combined Conditioning")
    print("=" * 70)
    
    batch_size = 4
    molecular_feature_dim = 128
    conditioning_dim = 256
    
    # Create individual conditioning modules
    mol_cond = MolecularConditioning(
        molecular_feature_dim=molecular_feature_dim,
        conditioning_dim=conditioning_dim,
        use_geometry=False
    )
    
    prop_cond = PropertyConditioning(
        property_names=['bandgap', 'melting_point'],
        conditioning_dim=conditioning_dim
    )
    prop_cond.set_normalization_params(
        torch.tensor([2.5, 180.0]),
        torch.tensor([1.2, 50.0])
    )
    
    sg_embed = SpaceGroupEmbedding(embedding_dim=conditioning_dim)
    dens_cond = DensityConditioning(embedding_dim=conditioning_dim)
    
    # Create combined conditioning
    combined = ExtendedCombinedConditioning(
        molecular_conditioning=mol_cond,
        space_group_embedding=sg_embed,
        density_conditioning=dens_cond,
        property_conditioning=prop_cond,
        conditioning_dim=conditioning_dim
    )
    
    # Example inputs
    mol_features = {
        'global_features': torch.randn(batch_size, molecular_feature_dim)
    }
    space_group = torch.tensor([14, 15, 14, 19])  # P21/c, C2/c, P21/c, P212121
    density = torch.tensor([1.2, 1.3, 1.1, 1.4])
    properties = torch.tensor([
        [2.5, 180.0],
        [3.0, 200.0],
        [2.8, 190.0],
        [2.3, 170.0],
    ])
    
    # Get combined conditioning
    combined_conditioning = combined(
        mol_features,
        space_group=space_group,
        density=density,
        properties=properties
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Number of conditioning types: {combined.num_conditionings}")
    print(f"Combined conditioning shape: {combined_conditioning.shape}")
    print()
    
    # Example with only molecular and property conditioning
    print("With only molecular + property conditioning:")
    combined_conditioning_partial = combined(
        mol_features,
        properties=properties
    )
    print(f"Partial conditioning shape: {combined_conditioning_partial.shape}")
    print()


def example_dataset_workflow():
    """Example 4: Dataset with properties workflow."""
    print("=" * 70)
    print("Example 4: Dataset Workflow (Conceptual)")
    print("=" * 70)
    
    print("""
    # Step 1: Create dataset with properties
    from crystal.data.crystal_loader import CrystalDatasetWithProperties
    
    dataset = CrystalDatasetWithProperties(
        db_path='data/crystals_with_props.db',
        indices=list(range(1000)),
        property_names=['bandgap', 'melting_point', 'dielectric_constant']
    )
    
    # Step 2: Get normalization statistics
    print(f"Property mean: {dataset.property_mean}")
    print(f"Property std: {dataset.property_std}")
    
    # Step 3: Set up property conditioning
    prop_cond = PropertyConditioning(
        property_names=dataset.property_names,
        conditioning_dim=256
    )
    prop_cond.set_normalization_params(
        dataset.property_mean,
        dataset.property_std
    )
    
    # Step 4: Training loop (conceptual)
    for batch in dataloader:
        molecular_features = batch['molecular_features']
        properties = batch['properties']  # From dataset
        space_group = batch['space_group']
        density = batch['density']
        
        # Get combined conditioning
        conditioning = combined_conditioning(
            molecular_features,
            space_group=space_group,
            density=density,
            properties=properties
        )
        
        # Use conditioning for crystal generation...
    
    # Step 5: Generation with target properties
    target_properties = torch.tensor([[2.5, 180.0, 3.0]])
    conditioning = prop_cond(target_properties)
    # Generate crystal with this conditioning...
    """)
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Property-Conditioned Crystal Generation Examples")
    print("=" * 70 + "\n")
    
    example_property_conditioning()
    example_property_normalizer()
    example_combined_conditioning()
    example_dataset_workflow()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
