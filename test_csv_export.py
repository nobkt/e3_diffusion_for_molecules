#!/usr/bin/env python3
"""
Unit tests for CSV export functionality (without requiring full dependencies)

This test validates:
1. Code syntax and structure
2. Function signatures
3. Argument parsing
"""

import sys
import os
import ast
import re

def test_main_qm9_arguments():
    """Test that main_qm9.py has the correct argument."""
    print("Testing main_qm9.py arguments...")
    
    with open('main_qm9.py', 'r') as f:
        content = f.read()
    
    # Check for export_conditions_csv argument
    if '--export_conditions_csv' in content:
        print("  ✓ --export_conditions_csv argument found")
    else:
        print("  ✗ --export_conditions_csv argument not found")
        return False
    
    # Check for help text
    if 'Export generation conditions' in content:
        print("  ✓ Help text found")
    else:
        print("  ✗ Help text not found")
        return False
    
    # Check for CSV export logic
    if 'if args.export_conditions_csv is not None:' in content:
        print("  ✓ CSV export condition check found")
    else:
        print("  ✗ CSV export condition check not found")
        return False
    
    # Check for exit after export
    if 'exit(0)' in content:
        print("  ✓ Exit statement found")
    else:
        print("  ✗ Exit statement not found")
        return False
    
    return True

def test_dataset_export_function():
    """Test that qm9/dataset.py has the export function."""
    print("\nTesting qm9/dataset.py export function...")
    
    with open('qm9/dataset.py', 'r') as f:
        content = f.read()
    
    # Check for function definition
    if 'def export_generation_conditions_to_csv' in content:
        print("  ✓ export_generation_conditions_to_csv function found")
    else:
        print("  ✗ export_generation_conditions_to_csv function not found")
        return False
    
    # Check for required parameters
    if 'datasets, output_dir' in content:
        print("  ✓ Required parameters found")
    else:
        print("  ✗ Required parameters not found")
        return False
    
    # Check for CSV file generation
    csv_files = [
        'molecular_weight.csv',
        'pi_conjugation_ratio.csv',
        'atom_types_encoding.csv',
        'functional_groups_encoding.csv'
    ]
    
    all_found = True
    for csv_file in csv_files:
        if csv_file in content:
            print(f"  ✓ {csv_file} generation found")
        else:
            print(f"  ✗ {csv_file} generation not found")
            all_found = False
    
    return all_found

def test_csv_headers():
    """Test that CSV headers are correctly defined."""
    print("\nTesting CSV headers...")
    
    with open('qm9/dataset.py', 'r') as f:
        content = f.read()
    
    # Check for Japanese headers
    japanese_headers = ['分子の組成', '分子量', 'π共役比率']
    all_found = True
    
    for header in japanese_headers:
        if header in content:
            print(f"  ✓ '{header}' header found")
        else:
            print(f"  ✗ '{header}' header not found")
            all_found = False
    
    return all_found

def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")
    
    files = ['CSV_EXPORT_USAGE.md', 'csv_export_example.py']
    all_found = True
    
    for filename in files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✓ {filename} exists ({size} bytes)")
        else:
            print(f"  ✗ {filename} not found")
            all_found = False
    
    return all_found

def test_syntax():
    """Test Python syntax of modified files."""
    print("\nTesting Python syntax...")
    
    files = ['main_qm9.py', 'qm9/dataset.py', 'csv_export_example.py']
    all_valid = True
    
    for filename in files:
        try:
            with open(filename, 'r') as f:
                ast.parse(f.read())
            print(f"  ✓ {filename} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {filename} has syntax error: {e}")
            all_valid = False
    
    return all_valid

def main():
    print("="*60)
    print("CSV Export Feature Validation Tests")
    print("="*60)
    
    # Change to repository directory
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_dir)
    
    tests = [
        ("Syntax", test_syntax),
        ("Main arguments", test_main_qm9_arguments),
        ("Export function", test_dataset_export_function),
        ("CSV headers", test_csv_headers),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
