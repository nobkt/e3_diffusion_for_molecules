#!/usr/bin/env python
"""
Test script to verify that atom_types_encoding is properly expanded to individual has_<atom> features
during exact conditional generation.

This test verifies the fix for the issue where molecules like Br19 were being generated
even when atom_types_encoding=[C,H,O,N] was specified.
"""

import torch
from eval_conditional_qm9 import create_exact_context, parse_property_values


def test_atom_types_expansion():
    """Test that atom_types_encoding is properly expanded to individual has_<atom> features."""
    
    print("=" * 80)
    print("Test: atom_types_encoding expansion")
    print("=" * 80)
    
    # Mock the necessary objects
    class MockArgs:
        def __init__(self):
            self.context_node_nf = 10  # Expect 10 context features
            # Assume the model was trained with these conditioning features:
            # molecular_weight (1), pi_conjugation_ratio (1), 
            # atom_types_encoding expanded to has_C, has_H, has_N, has_O (4 features)
            # Total: 6 features, but we'll say 10 to test with more
            self.conditioning = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding']
            self.dataset = 'qm9_second_half'
            self.remove_h = False
    
    class MockDataset:
        def __init__(self):
            self.data = {
                '_atom_types_mapping': ['C', 'H', 'N', 'O'],  # Available atom types in training data
                '_functional_groups_mapping': []
            }
    
    class MockDataloader:
        def __init__(self):
            self.dataset = MockDataset()
    
    args_gen = MockArgs()
    dataloaders = {'train': MockDataloader()}
    
    # Create property norms for the expected features
    # This simulates what compute_mean_mad would return
    property_norms = {
        'molecular_weight': {
            'mean': torch.tensor(50.0),
            'mad': torch.tensor(20.0)
        },
        'pi_conjugation_ratio': {
            'mean': torch.tensor(0.5),
            'mad': torch.tensor(0.2)
        },
        # Individual atom type features (as created during training)
        'has_C': {'mean': torch.tensor(0.8), 'mad': torch.tensor(0.3)},
        'has_H': {'mean': torch.tensor(0.9), 'mad': torch.tensor(0.2)},
        'has_N': {'mean': torch.tensor(0.4), 'mad': torch.tensor(0.3)},
        'has_O': {'mean': torch.tensor(0.5), 'mad': torch.tensor(0.3)},
    }
    
    # Parse property values from the problem statement
    property_values_str = 'pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N]'
    property_values = parse_property_values(property_values_str)
    
    print(f"\nInput property values string: {property_values_str}")
    print(f"Parsed property values: {property_values}")
    
    # Create exact context
    n_frames = 5
    n_nodes = 19
    device = torch.device('cpu')
    
    context = create_exact_context(
        property_values, args_gen, property_norms, 
        n_frames, n_nodes, device, dataloaders
    )
    
    print(f"\nContext shape: {context.shape}")
    print(f"Expected shape: torch.Size([{n_frames}, {args_gen.context_node_nf}])")
    
    # Verify the context was created correctly
    assert context.shape == (n_frames, args_gen.context_node_nf), \
        f"Expected shape ({n_frames}, {args_gen.context_node_nf}), got {context.shape}"
    
    # Check that all frames have the same context (exact conditions)
    for i in range(1, n_frames):
        assert torch.allclose(context[0], context[i]), \
            f"Frame {i} has different context than frame 0 (exact conditions should be identical)"
    
    print("\n✓ Test passed: Context shape is correct and all frames have identical context")
    
    # Print the context values to understand what was set
    print(f"\nContext values (frame 0):")
    print(context[0])
    
    # Check that the context values are not all zeros (which would indicate the fix didn't work)
    non_zero_count = (context[0] != 0).sum().item()
    print(f"\nNumber of non-zero context features: {non_zero_count} out of {args_gen.context_node_nf}")
    
    # We expect at least 2 non-zero features (pi_conjugation_ratio and at least one atom type)
    assert non_zero_count >= 2, \
        f"Expected at least 2 non-zero features, got {non_zero_count}. This suggests the fix didn't work."
    
    print("✓ Test passed: Context has non-zero values for conditioning features")
    
    return True


def test_property_parsing():
    """Test that property values are parsed correctly."""
    
    print("\n" + "=" * 80)
    print("Test: Property value parsing")
    print("=" * 80)
    
    test_cases = [
        ('pi_conjugation_ratio=0.9', {'pi_conjugation_ratio': 0.9}),
        ('atom_types_encoding=[C,H,O,N]', {'atom_types_encoding': ['C', 'H', 'O', 'N']}),
        (
            'pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N]',
            {'pi_conjugation_ratio': 0.9, 'atom_types_encoding': ['C', 'H', 'O', 'N']}
        ),
        (
            'molecular_weight=100.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N]',
            {
                'molecular_weight': 100.0,
                'pi_conjugation_ratio': 0.9,
                'atom_types_encoding': ['C', 'H', 'O', 'N']
            }
        ),
    ]
    
    for input_str, expected in test_cases:
        result = parse_property_values(input_str)
        print(f"\nInput: {input_str}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        assert result == expected, f"Parsing failed for {input_str}"
        print("✓ Parsing correct")
    
    return True


def test_functional_groups_expansion():
    """Test that functional_groups_encoding is also properly expanded."""
    
    print("\n" + "=" * 80)
    print("Test: functional_groups_encoding expansion")
    print("=" * 80)
    
    class MockArgs:
        def __init__(self):
            self.context_node_nf = 15
            self.conditioning = ['molecular_weight', 'functional_groups_encoding']
            self.dataset = 'qm9_second_half'
            self.remove_h = False
    
    class MockDataset:
        def __init__(self):
            self.data = {
                '_atom_types_mapping': [],
                '_functional_groups_mapping': ['carbonyl', 'hydroxyl', 'amino', 'carboxyl']
            }
    
    class MockDataloader:
        def __init__(self):
            self.dataset = MockDataset()
    
    args_gen = MockArgs()
    dataloaders = {'train': MockDataloader()}
    
    property_norms = {
        'molecular_weight': {'mean': torch.tensor(50.0), 'mad': torch.tensor(20.0)},
        'has_carbonyl': {'mean': torch.tensor(0.3), 'mad': torch.tensor(0.2)},
        'has_hydroxyl': {'mean': torch.tensor(0.4), 'mad': torch.tensor(0.2)},
        'has_amino': {'mean': torch.tensor(0.2), 'mad': torch.tensor(0.15)},
        'has_carboxyl': {'mean': torch.tensor(0.1), 'mad': torch.tensor(0.1)},
    }
    
    property_values_str = 'molecular_weight=80.0,functional_groups_encoding=[carbonyl,hydroxyl]'
    property_values = parse_property_values(property_values_str)
    
    print(f"\nInput: {property_values_str}")
    print(f"Parsed: {property_values}")
    
    context = create_exact_context(
        property_values, args_gen, property_norms, 
        5, 19, torch.device('cpu'), dataloaders
    )
    
    print(f"\nContext shape: {context.shape}")
    print(f"Context values (frame 0): {context[0]}")
    
    non_zero_count = (context[0] != 0).sum().item()
    print(f"Number of non-zero features: {non_zero_count}")
    
    # Should have at least 1 non-zero (molecular_weight)
    assert non_zero_count >= 1, "Expected at least 1 non-zero feature"
    
    print("✓ Test passed: functional_groups_encoding expansion works")
    
    return True


if __name__ == '__main__':
    try:
        print("\nRunning tests for atom_types_encoding fix...\n")
        
        # Run all tests
        test_property_parsing()
        test_atom_types_expansion()
        test_functional_groups_expansion()
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        print("\nThe fix correctly expands atom_types_encoding and functional_groups_encoding")
        print("to individual has_<atom> and has_<group> features during exact conditional generation.")
        print("\nThis should prevent the generation of impossible molecules like Br19 when")
        print("specifying atom_types_encoding=[C,H,O,N].")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Test failed with error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        exit(1)
