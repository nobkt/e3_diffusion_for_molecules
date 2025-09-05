#!/usr/bin/env python3
"""
Test script for exact conditional generation.
This script tests the parsing and validation of exact property conditions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eval_conditional_qm9 import parse_property_values, parse_single_property_value

def test_property_parsing():
    """Test the property value parsing functions."""
    
    print("Testing property value parsing...")
    
    # Test 1: Simple scalar values
    test_str1 = "molecular_weight=50.0,pi_conjugation_ratio=0.9"
    parsed1 = parse_property_values(test_str1)
    print(f"Test 1 input: {test_str1}")
    print(f"Test 1 output: {parsed1}")
    assert parsed1['molecular_weight'] == 50.0
    assert parsed1['pi_conjugation_ratio'] == 0.9
    print("✓ Test 1 passed\n")
    
    # Test 2: Atom types encoding
    test_str2 = "atom_types_encoding=[C,H,N,O]"
    parsed2 = parse_property_values(test_str2)
    print(f"Test 2 input: {test_str2}")
    print(f"Test 2 output: {parsed2}")
    assert parsed2['atom_types_encoding'] == ['C', 'H', 'N', 'O']
    print("✓ Test 2 passed\n")
    
    # Test 3: Functional groups encoding (simplified)
    test_str3 = "functional_groups_encoding=[carbonyl,hydroxyl]"
    parsed3 = parse_property_values(test_str3)
    print(f"Test 3 input: {test_str3}")
    print(f"Test 3 output: {parsed3}")
    assert parsed3['functional_groups_encoding'] == ['carbonyl', 'hydroxyl']
    print("✓ Test 3 passed\n")
    
    # Test 4: Combined properties
    test_str4 = "molecular_weight=75.5,pi_conjugation_ratio=0.6,atom_types_encoding=[C,H,O]"
    parsed4 = parse_property_values(test_str4)
    print(f"Test 4 input: {test_str4}")
    print(f"Test 4 output: {parsed4}")
    assert parsed4['molecular_weight'] == 75.5
    assert parsed4['pi_conjugation_ratio'] == 0.6
    assert parsed4['atom_types_encoding'] == ['C', 'H', 'O']
    print("✓ Test 4 passed\n")
    
    # Test 5: Single property value parsing
    print("Testing single property value parsing...")
    assert parse_single_property_value("50.0") == 50.0
    assert parse_single_property_value("[C,H,N]") == ['C', 'H', 'N']
    assert parse_single_property_value("0.9") == 0.9
    print("✓ Single value parsing tests passed\n")
    
    print("All property parsing tests passed! 🎉")

def test_example_from_problem_statement():
    """Test the exact example from the problem statement."""
    
    print("Testing example from problem statement...")
    
    # This is the example format from the problem statement (simplified for testing)
    example_str = "molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]"
    
    parsed = parse_property_values(example_str)
    print(f"Example input: {example_str}")
    print(f"Example output: {parsed}")
    
    # Validate the parsed results
    assert 'molecular_weight' in parsed
    assert 'pi_conjugation_ratio' in parsed
    assert 'atom_types_encoding' in parsed
    
    assert parsed['molecular_weight'] == 50.0
    assert parsed['pi_conjugation_ratio'] == 0.9
    assert parsed['atom_types_encoding'] == ['C', 'H', 'N', 'O']
    
    print("✓ Problem statement example parsing works correctly!")

def show_usage_examples():
    """Show usage examples for the new functionality."""
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Generate molecules with specific molecular weight:")
    print("python eval_conditional_qm9.py \\")
    print("  --generators_path outputs/your_model \\")
    print("  --task qualitative \\")
    print("  --use_exact_conditions \\")
    print("  --property_values 'molecular_weight=50.0'")
    
    print("\n2. Generate molecules with multiple exact conditions:")
    print("python eval_conditional_qm9.py \\")
    print("  --generators_path outputs/your_model \\")
    print("  --task qualitative \\")
    print("  --use_exact_conditions \\")
    print("  --property_values 'molecular_weight=50.0,pi_conjugation_ratio=0.9'")
    
    print("\n3. Generate molecules with specific atom types:")
    print("python eval_conditional_qm9.py \\")
    print("  --generators_path outputs/your_model \\")
    print("  --task qualitative \\")
    print("  --use_exact_conditions \\")
    print("  --property_values 'atom_types_encoding=[C,H,N,O]'")
    
    print("\n4. Generate molecules with all conditions (as in problem statement):")
    print("python eval_conditional_qm9.py \\")
    print("  --generators_path outputs/your_model \\")
    print("  --task qualitative \\")
    print("  --use_exact_conditions \\")
    print("  --property_values 'molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]'")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        test_property_parsing()
        test_example_from_problem_statement()
        show_usage_examples()
        print("\n🎉 All tests passed! The exact conditional generation is ready to use.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)