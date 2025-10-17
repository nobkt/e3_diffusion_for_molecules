#!/usr/bin/env python3
"""
Test script to verify that the circular import issue is fixed.
This test ensures that the modules can be imported without circular import errors.
"""

def test_import_egnn_models():
    """Test that egnn.models can be imported without errors."""
    print("Testing import of egnn.models...")
    try:
        import egnn.models
        print("✓ Successfully imported egnn.models")
        return True
    except AttributeError as e:
        print(f"✗ Failed to import egnn.models: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing egnn.models: {e}")
        return False


def test_import_en_diffusion():
    """Test that equivariant_diffusion.en_diffusion can be imported without errors."""
    print("Testing import of equivariant_diffusion.en_diffusion...")
    try:
        from equivariant_diffusion import en_diffusion
        print("✓ Successfully imported equivariant_diffusion.en_diffusion")
        return True
    except AttributeError as e:
        print(f"✗ Failed to import en_diffusion: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing en_diffusion: {e}")
        return False


def test_create_instances():
    """Test that we can create instances of the classes."""
    print("Testing instance creation...")
    try:
        from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
        from egnn.models import EGNN_dynamics_QM9
        import torch
        
        # Create dynamics instance
        dynamics = EGNN_dynamics_QM9(
            in_node_nf=5,
            context_node_nf=0,
            n_dims=3,
            hidden_nf=32,
            device='cpu'
        )
        print("✓ Successfully created EGNN_dynamics_QM9 instance")
        
        # Create EnVariationalDiffusion instance
        model = EnVariationalDiffusion(
            dynamics=dynamics,
            in_node_nf=5,
            n_dims=3,
            timesteps=10
        )
        print("✓ Successfully created EnVariationalDiffusion instance")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create instances: {e}")
        return False


def test_type_annotations():
    """Test that type annotations are preserved."""
    print("Testing type annotations...")
    try:
        from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
        
        annotations = EnVariationalDiffusion.__init__.__annotations__
        assert 'dynamics' in annotations, "dynamics parameter should have annotation"
        assert 'in_node_nf' in annotations, "in_node_nf parameter should have annotation"
        
        print(f"✓ Type annotations preserved: {annotations}")
        return True
    except Exception as e:
        print(f"✗ Failed to check type annotations: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Circular Import Fix")
    print("=" * 60)
    print()
    
    tests = [
        test_import_egnn_models,
        test_import_en_diffusion,
        test_create_instances,
        test_type_annotations,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed! 🎉")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
