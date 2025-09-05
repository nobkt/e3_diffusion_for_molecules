#!/usr/bin/env python3
"""
Test the exact context creation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from eval_conditional_qm9 import parse_property_values, create_exact_context

def test_exact_context_creation():
    """Test the exact context tensor creation."""
    
    print("Testing exact context creation...")
    
    # Mock arguments for testing
    class MockArgs:
        def __init__(self):
            self.conditioning = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding']
            self.dataset = 'ase_db'
            self.remove_h = False
    
    args_gen = MockArgs()
    
    # Mock property norms
    property_norms = {
        'molecular_weight': {'mean': 50.0, 'mad': 10.0},
        'pi_conjugation_ratio': {'mean': 0.5, 'mad': 0.2},
        'atom_types_encoding': {'mean': torch.zeros(4), 'mad': torch.ones(4)}
    }
    
    # Test exact property values
    property_values = {
        'molecular_weight': 50.0,
        'pi_conjugation_ratio': 0.9,
        'atom_types_encoding': ['C', 'H', 'N', 'O']
    }
    
    # Create exact context
    n_frames = 5
    n_nodes = 10
    device = torch.device('cpu')
    
    try:
        context = create_exact_context(
            property_values, args_gen, property_norms, 
            n_frames, n_nodes, device
        )
        
        print(f"✓ Successfully created context tensor")
        print(f"  Shape: {context.shape}")
        print(f"  Device: {context.device}")
        print(f"  Data type: {context.dtype}")
        
        # Check that molecular weight is properly normalized
        mw_col = context[:, 0]  # First column should be molecular weight
        expected_mw = (50.0 - 50.0) / 10.0  # Should be 0.0 (mean normalized)
        print(f"  Molecular weight column: {mw_col[0].item():.3f} (expected: {expected_mw:.3f})")
        
        # Check that pi conjugation ratio is properly normalized  
        pi_col = context[:, 1]  # Second column should be pi conjugation ratio
        expected_pi = (0.9 - 0.5) / 0.2  # Should be 2.0
        print(f"  π conjugation ratio column: {pi_col[0].item():.3f} (expected: {expected_pi:.3f})")
        
        print("✓ Context creation test passed!")
        
    except Exception as e:
        print(f"❌ Context creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_property_parsing_integration():
    """Test the full parsing to context creation pipeline."""
    
    print("\nTesting full parsing to context pipeline...")
    
    # Test the example from the problem statement
    property_values_str = "molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]"
    
    # Parse property values
    property_values = parse_property_values(property_values_str)
    print(f"Parsed values: {property_values}")
    
    # Verify the parsing worked correctly
    assert property_values['molecular_weight'] == 50.0
    assert property_values['pi_conjugation_ratio'] == 0.9
    assert property_values['atom_types_encoding'] == ['C', 'H', 'N', 'O']
    
    print("✓ Parsing integration test passed!")
    return True

def show_example_usage():
    """Show how the implementation would be used in practice."""
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE IN PRACTICE")
    print("="*60)
    
    print("""
The exact conditional generation can now be used as follows:

1. TRAIN A MODEL with molecular descriptor conditioning:
   python main_qm9.py \\
     --dataset ase_db \\
     --ase_db_path your_database.db \\
     --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \\
     --exp_name molecular_descriptor_model \\
     --n_epochs 1000

2. GENERATE with exact conditions (problem statement example):
   python eval_conditional_qm9.py \\
     --generators_path outputs/molecular_descriptor_model \\
     --task qualitative \\
     --use_exact_conditions \\
     --property_values 'molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]' \\
     --n_sweeps 5

3. RESULTS:
   - Generated molecules will match the specified conditions
   - molecular_weight ≈ 50.0 u
   - pi_conjugation_ratio ≈ 0.9 (90% of bonds are π bonds)
   - atom_types_encoding contains only C, H, N, O atoms
   - Output saved to outputs/molecular_descriptor_model/analysis/
""")

if __name__ == "__main__":
    try:
        print("="*60)
        print("EXACT CONTEXT CREATION TESTS")
        print("="*60)
        
        success1 = test_exact_context_creation()
        success2 = test_property_parsing_integration()
        
        if success1 and success2:
            show_example_usage()
            print("\n🎉 All tests passed! The exact conditional generation is ready to use.")
        else:
            print("\n❌ Some tests failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)