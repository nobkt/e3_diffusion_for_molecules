#!/usr/bin/env python3
"""
Test script for the automatic training optimization system.
Validates that the optimizer can detect issues and propose fixes.
"""

import sys
import os
sys.path.append(os.getcwd())

from training_optimizer import TrainingOptimizer
import torch

def test_high_loss_detection():
    """Test detection and optimization for high loss scenarios"""
    print("🧪 Testing high loss detection and optimization...")
    
    optimizer = TrainingOptimizer(initial_lr=1e-4, initial_norm_factors=[4.0, 4, 1])
    
    # Simulate high loss, low stability scenario
    distance_stats = {
        'mean': 7.2,
        'min': 0.2,
        'max': 23.0,
        'std': 3.5,
        'very_short_count': 23,
        'very_long_count': 6988
    }
    
    analysis = optimizer.analyze_training_state(
        loss=4.1,
        mol_stability=0.0,
        atm_stability=0.128,
        distance_stats=distance_stats,
        epoch=0
    )
    
    print(f"   Problems detected: {analysis['problems']}")
    print(f"   Critical: {analysis['critical']}")
    
    # Test optimization plan
    plan = optimizer.generate_optimization_plan(
        current_lr=1e-5,
        current_norm_factors=[4.267, 4, 1],
        analysis=analysis,
        loss=4.1,
        stability=0.0,
        distance_stats=distance_stats
    )
    
    print(f"   Apply fixes: {plan['apply_fixes']}")
    print(f"   New LR: {plan['new_lr']:.2e}")
    print(f"   New norm factors: {plan['new_norm_factors']}")
    
    # Test diffusion schedule analysis
    schedule_analysis = optimizer.analyze_diffusion_schedule_issues(4.1, 0.0, 0)
    print(f"   Schedule issues: {list(schedule_analysis.keys())}")
    
    assert analysis['critical'], "Should detect critical issues"
    assert plan['apply_fixes'], "Should recommend fixes"
    assert plan['new_lr'] < 1e-5, "Should reduce learning rate"
    print("   ✅ High loss detection test passed")

def test_normalization_optimization():
    """Test coordinate normalization optimization"""
    print("\n🧪 Testing coordinate normalization optimization...")
    
    optimizer = TrainingOptimizer(initial_lr=1e-4, initial_norm_factors=[4.0, 4, 1])
    
    # Simulate normalization issues with very bad distance distribution
    distance_stats = {
        'mean': 12.5,  # Too large
        'min': 0.1,
        'max': 30.0,
        'std': 8.5,
        'very_short_count': 100,  # Many short distances
        'very_long_count': 2000   # Many long distances
    }
    
    # Test normalization computation
    new_norm = optimizer.compute_optimal_coordinate_normalization(
        current_norm_factor=4.267,
        distance_stats=distance_stats,
        stability=0.05
    )
    
    print(f"   Original norm factor: 4.267")
    print(f"   Optimized norm factor: {new_norm:.3f}")
    
    # Should increase normalization factor to reduce coordinate scale
    assert new_norm > 4.267, "Should increase normalization factor for large distances"
    print("   ✅ Normalization optimization test passed")

def test_learning_rate_adjustment():
    """Test learning rate adjustment logic"""
    print("\n🧪 Testing learning rate adjustment...")
    
    optimizer = TrainingOptimizer(initial_lr=1e-4, initial_norm_factors=[4.0, 4, 1])
    
    # Simulate loss history
    optimizer.loss_history = [5.2, 4.8, 4.3, 4.1]  # Decreasing but high
    
    new_lr = optimizer.compute_optimal_learning_rate(
        current_lr=1e-5,
        loss=4.1,
        loss_history=optimizer.loss_history,
        stability=0.0
    )
    
    print(f"   Original LR: 1e-5")
    print(f"   Optimized LR: {new_lr:.2e}")
    
    # Should reduce learning rate due to high loss
    assert new_lr < 1e-5, "Should reduce learning rate for high loss"
    print("   ✅ Learning rate adjustment test passed")

def test_no_optimization_for_good_training():
    """Test that no optimization is applied when training is going well"""
    print("\n🧪 Testing no optimization for good training...")
    
    optimizer = TrainingOptimizer(initial_lr=1e-4, initial_norm_factors=[4.0, 4, 1])
    
    # Simulate good training state
    distance_stats = {
        'mean': 2.1,
        'min': 0.9,
        'max': 4.8,
        'std': 1.2,
        'very_short_count': 2,
        'very_long_count': 15
    }
    
    analysis = optimizer.analyze_training_state(
        loss=1.8,  # Reasonable loss
        mol_stability=0.85,  # Good stability
        atm_stability=0.92,
        distance_stats=distance_stats,
        epoch=0
    )
    
    plan = optimizer.generate_optimization_plan(
        current_lr=2e-4,
        current_norm_factors=[3.5, 4, 1],
        analysis=analysis,
        loss=1.8,
        stability=0.85,
        distance_stats=distance_stats
    )
    
    print(f"   Problems detected: {analysis['problems']}")
    print(f"   Apply fixes: {plan['apply_fixes']}")
    
    assert not analysis['critical'], "Should not detect critical issues"
    assert not plan['apply_fixes'], "Should not recommend fixes"
    print("   ✅ No optimization for good training test passed")

if __name__ == "__main__":
    print("🚀 Testing automatic training optimization system...")
    
    try:
        test_high_loss_detection()
        test_normalization_optimization()
        test_learning_rate_adjustment()
        test_no_optimization_for_good_training()
        
        print("\n🎉 All tests passed! Automatic optimization system is working correctly.")
        print("\nThe system can:")
        print("  ✓ Detect high loss and stability issues")
        print("  ✓ Automatically adjust learning rates")
        print("  ✓ Optimize coordinate normalization factors")
        print("  ✓ Provide diffusion schedule recommendations")
        print("  ✓ Only intervene when necessary")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)