#!/usr/bin/env python3
"""
Test script for numerical stability fixes.
Tests the key functions that were modified to ensure they handle edge cases properly.
"""

import torch
import numpy as np
from egnn.egnn_new import unsorted_segment_sum
from utils import gradient_clipping, Queue

def test_unsorted_segment_sum():
    """Test the enhanced unsorted_segment_sum function with various edge cases."""
    print("Testing unsorted_segment_sum improvements...")
    
    # Test 1: Normal case
    data = torch.randn(10, 5)
    segment_ids = torch.randint(0, 3, (10,))
    result = unsorted_segment_sum(data, segment_ids, 3, 100, 'sum')
    assert not torch.any(torch.isnan(result)), "Normal case should not produce NaN"
    assert not torch.any(torch.isinf(result)), "Normal case should not produce inf"
    print("✓ Normal case passed")
    
    # Test 2: Input with NaN
    data_nan = data.clone()
    data_nan[0, 0] = float('nan')
    result = unsorted_segment_sum(data_nan, segment_ids, 3, 100, 'sum')
    assert not torch.any(torch.isnan(result)), "Should handle NaN input"
    assert not torch.any(torch.isinf(result)), "Should handle NaN input"
    print("✓ NaN input handling passed")
    
    # Test 3: Input with inf
    data_inf = data.clone()
    data_inf[0, 0] = float('inf')
    result = unsorted_segment_sum(data_inf, segment_ids, 3, 100, 'sum')
    assert not torch.any(torch.isnan(result)), "Should handle inf input"
    assert not torch.any(torch.isinf(result)), "Should handle inf input"
    print("✓ Inf input handling passed")
    
    # Test 4: Very large values
    data_large = torch.ones(10, 5) * 1e10
    result = unsorted_segment_sum(data_large, segment_ids, 3, 100, 'sum')
    assert not torch.any(torch.isnan(result)), "Should handle large values"
    assert not torch.any(torch.isinf(result)), "Should handle large values"
    assert torch.all(torch.abs(result) <= 1e6), "Should clip extreme values"
    print("✓ Large value handling passed")
    
    # Test 5: Small normalization factor
    result = unsorted_segment_sum(data, segment_ids, 3, 1e-15, 'sum')
    assert not torch.any(torch.isnan(result)), "Should handle small normalization factor"
    assert not torch.any(torch.isinf(result)), "Should handle small normalization factor"
    print("✓ Small normalization factor handling passed")
    
    # Test 6: Mean aggregation
    result = unsorted_segment_sum(data, segment_ids, 3, 100, 'mean')
    assert not torch.any(torch.isnan(result)), "Mean aggregation should work"
    assert not torch.any(torch.isinf(result)), "Mean aggregation should work"
    print("✓ Mean aggregation passed")
    
    print("All unsorted_segment_sum tests passed!\n")

def test_gradient_clipping():
    """Test the enhanced gradient clipping function."""
    print("Testing gradient clipping improvements...")
    
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    
    # Test 1: Normal gradients
    queue = Queue()
    queue.add(100.0)
    queue.add(150.0)
    queue.add(120.0)
    
    # Set normal gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1
    
    grad_norm = gradient_clipping(model, queue)
    assert not np.isnan(grad_norm), "Normal gradients should not produce NaN"
    assert not np.isinf(grad_norm), "Normal gradients should not produce inf"
    print("✓ Normal gradient case passed")
    
    # Test 2: NaN gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1
        p.grad[0, 0] = float('nan')
    
    grad_norm = gradient_clipping(model, queue)
    # Check that gradients were fixed
    for p in model.parameters():
        assert not torch.any(torch.isnan(p.grad)), "NaN gradients should be fixed"
    print("✓ NaN gradient handling passed")
    
    # Test 3: Very large gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e6
    
    grad_norm = gradient_clipping(model, queue)
    # Should use conservative clipping
    assert grad_norm <= 1000.0, "Should apply conservative clipping for large gradients"
    print("✓ Large gradient handling passed")
    
    # Test 4: Queue with extreme values
    extreme_queue = Queue()
    extreme_queue.add(1e8)
    extreme_queue.add(1e7)
    extreme_queue.add(1e6)
    
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1
    
    grad_norm = gradient_clipping(model, extreme_queue)
    # Queue should be reset to reasonable values
    assert extreme_queue.mean() <= 1e3, "Extreme queue should be reset"
    print("✓ Extreme queue handling passed")
    
    print("All gradient clipping tests passed!\n")

def test_integration():
    """Test that the fixes work together in a simple integration test."""
    print("Testing integration of fixes...")
    
    # Simulate a simple forward pass with potential numerical issues
    batch_size, n_nodes, n_dims = 2, 10, 3
    
    # Create data that might cause issues
    data = torch.randn(batch_size * n_nodes, 5) * 1e3  # Large scale data
    segment_ids = torch.randint(0, batch_size * n_nodes, (batch_size * n_nodes,))
    
    # Test the aggregation function
    result = unsorted_segment_sum(data, segment_ids, batch_size * n_nodes, 100, 'sum')
    
    assert not torch.any(torch.isnan(result)), "Integration test: should handle large scale data"
    assert not torch.any(torch.isinf(result)), "Integration test: should handle large scale data"
    assert torch.all(torch.abs(result) <= 1e6), "Integration test: should clip extreme values"
    
    print("✓ Integration test passed")
    print("All tests completed successfully! 🎉")

if __name__ == "__main__":
    test_unsorted_segment_sum()
    test_gradient_clipping()
    test_integration()