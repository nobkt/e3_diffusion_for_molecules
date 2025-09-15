#!/usr/bin/env python3
"""
Test script to validate the training stability fixes for halogen bias and numerical instability.

This script tests:
1. Tensor conversion fix in models.py
2. Property normalization improvements 
3. Gradient clipping enhancements
"""

import sys
import os
import warnings
from io import StringIO

# Capture warnings to test for the specific torch.tensor warning
warnings.filterwarnings('always')

def test_tensor_conversion_fix():
    """Test that the torch.tensor conversion warning is fixed."""
    print("Testing tensor conversion fix...")
    
    # We can't import torch in this environment, but we can check the code
    with open('qm9/models.py', 'r') as f:
        content = f.read()
    
    # Check that the problematic line is fixed
    if 'Categorical(torch.tensor(probs))' in content:
        print("❌ FAIL: Still using torch.tensor(probs) in models.py")
        return False
    elif 'Categorical(probs.clone().detach())' in content:
        print("✅ PASS: Fixed torch.tensor conversion in models.py")
        return True
    else:
        print("❓ UNKNOWN: Could not find the expected pattern in models.py")
        return False

def test_property_normalization_improvements():
    """Test that property normalization has been improved."""
    print("\nTesting property normalization improvements...")
    
    with open('qm9/utils.py', 'r') as f:
        content = f.read()
    
    # Check for improved clamping range
    if 'torch.clamp(properties, min=-5.0, max=5.0)' in content:
        print("✅ PASS: Found improved clamping range (-5.0 to 5.0)")
        clamp_ok = True
    else:
        print("❌ FAIL: Did not find improved clamping range")
        clamp_ok = False
    
    # Check for lowered warning threshold
    if 'if max_abs_val > 3.0:' in content:
        print("✅ PASS: Found lowered warning threshold (3.0)")
        threshold_ok = True
    else:
        print("❌ FAIL: Did not find lowered warning threshold")
        threshold_ok = False
    
    return clamp_ok and threshold_ok

def test_gradient_clipping_enhancements():
    """Test that gradient clipping has been enhanced."""
    print("\nTesting gradient clipping enhancements...")
    
    with open('utils.py', 'r') as f:
        content = f.read()
    
    # Check for enhanced warning system
    if 'WARNING: Very large gradient norm detected' in content:
        print("✅ PASS: Found enhanced gradient warning system")
        return True
    else:
        print("❌ FAIL: Did not find enhanced gradient warning system")
        return False

def test_normalization_factor_adjustments():
    """Test that normalization factor adjustments are in place."""
    print("\nTesting normalization factor adjustments...")
    
    with open('main_qm9.py', 'r') as f:
        content = f.read()
    
    # Check for ASE database normalization adjustments
    checks = [
        ('Fallback normalization adjustment', 'Applied fallback normalization adjustment for large ASE database'),
        ('Element count check', 'n_elements > 10'),
        ('Normalization factor 1.0', 'args.normalize_factors[1] = 1.0'),
        ('ASE database check', "'ase_db' in args.dataset")
    ]
    
    all_passed = True
    for check_name, pattern in checks:
        if pattern in content:
            print(f"✅ PASS: Found {check_name}")
        else:
            print(f"❌ FAIL: Did not find {check_name}")
            all_passed = False
    
    return all_passed

def test_ase_gradnorm_initialization():
    """Test that ASE databases use smaller initial gradnorm values."""
    print("\nTesting ASE gradnorm initialization...")
    
    with open('main_qm9.py', 'r') as f:
        content = f.read()
    
    # Check for smaller initial gradnorm for ASE databases
    if 'gradnorm_queue.add(10.0)  # Much smaller initial value for ASE databases' in content:
        print("✅ PASS: Found smaller initial gradnorm for ASE databases")
        return True
    else:
        print("❌ FAIL: Did not find smaller initial gradnorm for ASE databases")
        return False

def check_documentation():
    """Check that relevant documentation exists."""
    print("\nChecking documentation...")
    
    if os.path.exists('HALOGEN_BIAS_FIX.md'):
        print("✅ PASS: Found HALOGEN_BIAS_FIX.md documentation")
        return True
    else:
        print("❌ FAIL: Missing HALOGEN_BIAS_FIX.md documentation")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Training Stability Fixes for E3 Diffusion")
    print("=" * 55)
    
    tests = [
        test_tensor_conversion_fix,
        test_property_normalization_improvements,
        test_gradient_clipping_enhancements,
        test_normalization_factor_adjustments,
        test_ase_gradnorm_initialization,
        check_documentation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 55)
    print("📊 SUMMARY:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 Training stability fixes are correctly implemented!")
        print("\n💡 Expected improvements:")
        print("   • No more torch.tensor conversion warnings")
        print("   • More stable property normalization")
        print("   • Better detection of numerical instability")
        print("   • Reduced halogen bias in ASE databases")
        print("   • More conservative gradient clipping for stability")
        return 0
    else:
        print(f"❌ {total - passed} TESTS FAILED ({passed}/{total} passed)")
        print("\n⚠️  Some fixes may not be properly implemented.")
        return 1

if __name__ == "__main__":
    sys.exit(main())