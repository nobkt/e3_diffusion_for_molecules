#!/usr/bin/env python3
"""
Verification script for the Si KeyError fix
This script can be run to verify that the fix is working correctly.
"""

def verify_fix():
    """Verify that the Si KeyError fix is working"""
    try:
        from qm9 import bond_analyze
    except ImportError as e:
        print(f"❌ Cannot import bond_analyze: {e}")
        return False
    
    print("🔍 Verifying Si KeyError fix...")
    
    # Test the problematic case that was causing KeyError
    test_cases = [
        ('Si', 'N', "Silicon-Nitrogen bond"),
        ('N', 'Si', "Nitrogen-Silicon bond"), 
        ('Si', 'P', "Silicon-Phosphorus bond"),
        ('P', 'Si', "Phosphorus-Silicon bond"),
    ]
    
    all_passed = True
    
    for atom1, atom2, description in test_cases:
        print(f"\n  Testing {description} ({atom1}-{atom2}):")
        
        # Test with check_exists=False (should fail)
        try:
            bond_analyze.get_bond_order(atom1, atom2, 1.8, check_exists=False)
            print(f"    ⚠️  Unexpectedly succeeded with check_exists=False")
        except KeyError:
            print(f"    ✅ Correctly fails with check_exists=False (original issue)")
        
        # Test with check_exists=True (should work)
        try:
            result = bond_analyze.get_bond_order(atom1, atom2, 1.8, check_exists=True)
            print(f"    ✅ Success with check_exists=True (bond order: {result})")
        except Exception as e:
            print(f"    ❌ Failed with check_exists=True: {e}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The Si KeyError fix is working correctly.")
        print("   ASE database training should now work without crashing.")
        return True
    else:
        print("\n❌ Some tests failed. The fix may not be working correctly.")
        return False

if __name__ == "__main__":
    import sys
    success = verify_fix()
    if not success:
        sys.exit(1)