#!/usr/bin/env python3
"""
Integration test that simulates the exact problem scenario described in the issue.
Tests the complete automatic optimization workflow.
"""

import sys
import os
sys.path.append(os.getcwd())

from training_optimizer import TrainingOptimizer

def test_problem_scenario():
    """Test the exact scenario described in the problem statement"""
    print("🧪 Testing the exact problem scenario from the issue...")
    
    # Initialize optimizer with the exact parameters from the problem
    training_optimizer = TrainingOptimizer(
        initial_lr=1e-5,  # From the command: --lr 1e-5
        initial_norm_factors=[4.267943859100342, 4, 1]  # From the output
    )
    
    # Simulate the exact conditions reported
    distance_stats = {
        'mean': 7.240,    # From: "Mean distance: 7.240"
        'min': 0.226,     # From: "Min distance: 0.226"
        'max': 23.114,    # From: "Max distance: 23.114"
        'std': 3.562,     # From: "Std distance: 3.562"
        'very_short_count': 23,    # From: "23 very short distances (<0.8 Å)"
        'very_long_count': 6988    # From: "6988 very long distances (>5.0 Å)"
    }
    
    # Loss around 4.0 as reported: "Loss 4.10, NLL: 4.10"
    loss = 4.10
    
    # Stability as reported: "Molecular stability: 0.000 (0.0%)" and "Atomic stability: 0.128 (12.8%)"
    mol_stability = 0.000
    atm_stability = 0.128
    
    print(f"   Simulating conditions:")
    print(f"     Loss: {loss}")
    print(f"     Molecular stability: {mol_stability} ({mol_stability*100:.1f}%)")
    print(f"     Atomic stability: {atm_stability} ({atm_stability*100:.1f}%)")
    print(f"     Distance mean: {distance_stats['mean']:.3f} Å")
    print(f"     Very short distances: {distance_stats['very_short_count']}")
    print(f"     Very long distances: {distance_stats['very_long_count']}")
    
    # Run the automatic analysis
    analysis = training_optimizer.analyze_training_state(
        loss=loss,
        mol_stability=mol_stability,
        atm_stability=atm_stability,
        distance_stats=distance_stats,
        epoch=0
    )
    
    print(f"\n   🔍 Analysis Results:")
    print(f"     Problems detected: {analysis['problems']}")
    print(f"     Critical issues: {analysis['critical']}")
    
    # Generate optimization plan
    optimization_plan = training_optimizer.generate_optimization_plan(
        current_lr=1e-5,
        current_norm_factors=[4.267943859100342, 4, 1],
        analysis=analysis,
        loss=loss,
        stability=mol_stability,
        distance_stats=distance_stats
    )
    
    print(f"\n   🔧 Optimization Plan:")
    print(f"     Apply fixes: {optimization_plan['apply_fixes']}")
    if optimization_plan['apply_fixes']:
        print(f"     New learning rate: {optimization_plan['new_lr']:.2e}")
        print(f"     New coord normalization: {optimization_plan['new_norm_factors'][0]:.3f}")
        print(f"     Rationale:")
        for reason in optimization_plan['rationale']:
            print(f"       • {reason}")
    
    # Test diffusion schedule recommendations
    schedule_analysis = training_optimizer.analyze_diffusion_schedule_issues(loss, mol_stability, 0)
    if schedule_analysis:
        print(f"\n   📋 Diffusion Schedule Recommendations:")
        for param, details in schedule_analysis.items():
            print(f"     {param}: {details['suggestion']}")
            print(f"       Reason: {details['rationale']}")
    
    # Validate that all critical issues are addressed
    assert analysis['critical'], "Should detect critical issues in this scenario"
    assert optimization_plan['apply_fixes'], "Should recommend fixes"
    assert 'high_loss' in analysis['problems'], "Should detect high loss"
    assert 'very_low_molecular_stability' in analysis['problems'], "Should detect low stability"
    assert 'coordinate_normalization_issue' in analysis['problems'], "Should detect distance issues"
    
    # Check that learning rate is reduced
    assert optimization_plan['new_lr'] < 1e-5, "Should reduce learning rate"
    
    # Check that coordinate normalization is adjusted for the distance issues
    original_norm = 4.267943859100342
    new_norm = optimization_plan['new_norm_factors'][0]
    assert abs(new_norm - original_norm) > 0.1, "Should significantly adjust normalization"
    
    print(f"\n   ✅ All validations passed!")
    
    # Simulate what the user would see
    print(f"\n🚀 What the user would see during training:")
    print(f"🤖 AUTOMATIC OPTIMIZATION ANALYSIS:")
    print(f"   Problems detected: {', '.join(analysis['problems'])}")
    print(f"   Critical issues: {'Yes' if analysis['critical'] else 'No'}")
    print(f"")
    print(f"🔧 APPLYING AUTOMATIC OPTIMIZATIONS:")
    print(f"   ✓ Learning rate: {1e-5:.2e} → {optimization_plan['new_lr']:.2e}")
    print(f"   ✓ Coordinate normalization: {original_norm:.3f} → {new_norm:.3f}")
    print(f"   Rationale:")
    for reason in optimization_plan['rationale']:
        print(f"     • {reason}")
    print(f"   Expected improvements:")
    for improvement in optimization_plan['expected_improvements']:
        print(f"     • {improvement}")
    
    if schedule_analysis:
        print(f"")
        print(f"📋 DIFFUSION SCHEDULE ANALYSIS:")
        for param, details in schedule_analysis.items():
            print(f"   {param.upper()} ISSUE: {details['issue']}")
            print(f"     Suggestion: {details['suggestion']}")
            print(f"     ⚠️  Manual adjustment required - restart training with suggested parameters")

if __name__ == "__main__":
    print("🎯 Testing automatic optimization with the exact problem scenario...")
    print("This simulates the conditions reported in the GitHub issue.")
    
    try:
        test_problem_scenario()
        
        print(f"\n🎉 SUCCESS: The automatic optimization system correctly handles the reported problem!")
        print(f"\nKey improvements:")
        print(f"  ✓ Detects all critical issues (high loss, low stability, distance problems)")
        print(f"  ✓ Automatically reduces learning rate to address convergence issues")
        print(f"  ✓ Adjusts coordinate normalization to fix distance distribution")
        print(f"  ✓ Provides diffusion schedule recommendations for manual adjustment")
        print(f"  ✓ Gives clear rationale and expected improvements")
        print(f"\nThe user no longer needs to manually diagnose and fix these issues!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)