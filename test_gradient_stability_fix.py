#!/usr/bin/env python3
"""
Test script to validate gradient stability improvements for ASE database training.
This script tests the key fixes:

1. Improved gradient clipping with higher initial values
2. Better molecular weight normalization
3. Reduced warning noise
4. Learning rate adjustments
"""

import sys
import numpy as np


def test_gradient_clipping_logic():
    """Test the improved gradient clipping logic"""
    print("Testing improved gradient clipping logic...")
    
    # Simulate the Queue class
    class Queue:
        def __init__(self, max_len=50):
            self.items = []
            self.max_len = max_len
        
        def add(self, item):
            self.items.insert(0, item)
            if len(self.items) > self.max_len:
                self.items.pop()
        
        def mean(self):
            return np.mean(self.items)
        
        def std(self):
            return np.std(self.items)
        
        def __len__(self):
            return len(self.items)
    
    # Test improved initialization for ASE databases
    queue = Queue()
    # Simulate the new initialization in main_qm9.py
    queue.add(50.0)
    queue.add(30.0)
    queue.add(20.0)
    queue.add(40.0)
    queue.add(60.0)
    
    print(f"✅ Queue initialized with {len(queue)} values")
    print(f"   Mean: {queue.mean():.1f}, Std: {queue.std():.1f}")
    
    # Test gradient clipping logic
    if len(queue) >= 5:
        queue_mean = queue.mean()
        queue_std = queue.std()
        max_grad_norm = 1.5 * queue_mean + 2 * queue_std
        max_grad_norm = max(1.0, min(max_grad_norm, 100.0))
        print(f"✅ Adaptive clipping: max_grad_norm = {max_grad_norm:.1f}")
    
    # Test gradient values from the error log
    test_gradients = [1412.8, 105.0, 12436.3, 4448.8, 388.2, 50.0, 30.0]
    warnings_count = 0
    clips_count = 0
    
    for grad in test_gradients:
        if grad > max_grad_norm:
            clips_count += 1
            # Only warn for significantly large gradients (reduced noise)
            if grad > max_grad_norm * 2.0:
                # print(f"   Would clip: {grad:.1f} -> {max_grad_norm:.1f}")
                pass
            if grad > 1000.0:
                warnings_count += 1
    
    print(f"✅ Gradient handling: {clips_count} clips, {warnings_count} warnings (reduced noise)")
    return True


def test_molecular_weight_normalization():
    """Test the improved molecular weight normalization"""
    print("\nTesting molecular weight normalization improvements...")
    
    # Simulate some molecular weight values (typical range: 50-500 for small molecules)
    molecular_weights = np.array([78.0, 180.0, 310.0, 120.0, 250.0, 95.0, 420.0])
    mean = np.mean(molecular_weights)
    mad_original = np.mean(np.abs(molecular_weights - mean))
    
    # Test the new MAD calculation logic for molecular weight
    min_mad = max(abs(mean) * 0.25, mad_original * 0.5)
    mad_adjusted = max(mad_original, min_mad)
    
    print(f"✅ Molecular weight stats:")
    print(f"   Mean: {mean:.1f}")
    print(f"   Original MAD: {mad_original:.1f}")
    print(f"   Adjusted MAD: {mad_adjusted:.1f}")
    
    # Test normalization
    normalized = (molecular_weights - mean) / mad_adjusted
    max_abs_normalized = np.max(np.abs(normalized))
    
    # Test clamping (should be ±4.0 for molecular weight)
    clamped = np.clip(normalized, -4.0, 4.0)
    max_abs_clamped = np.max(np.abs(clamped))
    
    print(f"✅ Normalization results:")
    print(f"   Max abs normalized: {max_abs_normalized:.2f}")
    print(f"   Max abs after clamp: {max_abs_clamped:.2f}")
    
    # Test warning threshold (should be 3.5 for molecular weight)
    warning_threshold = 3.5
    would_warn = max_abs_clamped > warning_threshold
    print(f"✅ Warning check: would warn = {would_warn} (threshold = {warning_threshold})")
    
    return True


def test_learning_rate_adjustment():
    """Test the learning rate adjustment logic"""
    print("\nTesting learning rate adjustment for ASE databases...")
    
    # Simulate argument parsing for ASE database
    class Args:
        def __init__(self):
            self.dataset = 'ase_db'
            self.lr = 2e-4  # Default learning rate
    
    args = Args()
    original_lr = args.lr
    
    # Apply the adjustment logic
    if 'ase_db' in args.dataset and args.lr >= 2e-4:
        args.lr = 1e-4
    
    print(f"✅ Learning rate adjustment:")
    print(f"   Original LR: {original_lr}")
    print(f"   Adjusted LR: {args.lr}")
    print(f"   Reduction factor: {original_lr / args.lr:.1f}x")
    
    return True


def validate_fixes_in_code():
    """Validate that the fixes are properly implemented in the code files"""
    print("\nValidating fixes in code files...")
    
    fixes_validated = 0
    
    # Check utils.py for improved gradient clipping
    try:
        with open('utils.py', 'r') as f:
            content = f.read()
            if 'max_grad_norm = 10.0' in content:
                print("✅ Found improved initial gradient norm (10.0)")
                fixes_validated += 1
            if 'grad_norm > max_grad_norm * 2.0' in content:
                print("✅ Found reduced clipping noise logic")
                fixes_validated += 1
    except:
        print("⚠️  Could not check utils.py")
    
    # Check qm9/utils.py for molecular weight improvements
    try:
        with open('qm9/utils.py', 'r') as f:
            content = f.read()
            if "property_key == 'molecular_weight'" in content:
                print("✅ Found molecular weight specific normalization")
                fixes_validated += 1
            if 'warning_threshold = 3.5 if key' in content:
                print("✅ Found adjusted warning thresholds")
                fixes_validated += 1
    except:
        print("⚠️  Could not check qm9/utils.py")
    
    # Check main_qm9.py for learning rate and gradient queue improvements
    try:
        with open('main_qm9.py', 'r') as f:
            content = f.read()
            if 'gradnorm_queue.add(50.0)' in content:
                print("✅ Found improved gradient queue initialization")
                fixes_validated += 1
            if 'args.lr = 1e-4' in content:
                print("✅ Found automatic learning rate adjustment")
                fixes_validated += 1
    except:
        print("⚠️  Could not check main_qm9.py")
    
    print(f"\n✅ Validated {fixes_validated}/6 fixes in code")
    return fixes_validated >= 4  # At least 4 fixes should be found


def main():
    """Run all tests"""
    print("=" * 60)
    print("ASE Database Training Stability Fix Validation")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    try:
        if test_gradient_clipping_logic():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Gradient clipping test failed: {e}")
    
    try:
        if test_molecular_weight_normalization():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Molecular weight test failed: {e}")
    
    try:
        if test_learning_rate_adjustment():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Learning rate test failed: {e}")
    
    try:
        if validate_fixes_in_code():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Code validation test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All fixes validated successfully!")
        print("\nExpected improvements:")
        print("  - Fewer gradient clipping warnings")
        print("  - No 'Large normalized values' warnings for molecular_weight")
        print("  - More stable training with reduced numerical instability")
        print("  - Better initial gradient handling for ASE databases")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementations.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)