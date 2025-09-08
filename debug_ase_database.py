#!/usr/bin/env python3
"""
Debug script for diagnosing ASE database issues.

This script helps diagnose database corruption and compatibility issues
by testing various access methods and providing detailed error information.

Usage:
    python debug_ase_database.py path/to/database.db
"""

import sys
import os
import argparse
import traceback

def test_basic_file_access(db_path):
    """Test basic file system access."""
    print("🔍 BASIC FILE ACCESS TEST")
    print("-" * 40)
    
    try:
        # Check if file exists
        if not os.path.exists(db_path):
            print(f"❌ File does not exist: {db_path}")
            return False
        else:
            print(f"✅ File exists: {db_path}")
        
        # Check file size
        file_size = os.path.getsize(db_path)
        print(f"📏 File size: {file_size:,} bytes")
        
        if file_size == 0:
            print("❌ File is empty")
            return False
        elif file_size < 100:
            print("⚠️  File is very small, might be corrupted")
        
        # Check file permissions
        if os.access(db_path, os.R_OK):
            print("✅ File is readable")
        else:
            print("❌ File is not readable")
            return False
            
        # Try to read first few bytes
        try:
            with open(db_path, 'rb') as f:
                header = f.read(16)
                if header.startswith(b'SQLite format 3'):
                    print("✅ File has SQLite header")
                else:
                    print(f"⚠️  File doesn't start with SQLite header: {header[:16]}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Basic file access failed: {e}")
        return False

def test_sqlite_access(db_path):
    """Test SQLite database access."""
    print("\n🗄️  SQLITE ACCESS TEST")
    print("-" * 40)
    
    try:
        import sqlite3
        
        # Try to connect with sqlite3
        try:
            conn = sqlite3.connect(db_path)
            print("✅ SQLite connection successful")
            
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"📋 Tables found: {[table[0] for table in tables]}")
            
            # Check if it looks like an ASE database
            if any('systems' in str(table) for table in tables):
                print("✅ Appears to be an ASE database (has systems table)")
            else:
                print("⚠️  May not be an ASE database (no systems table found)")
            
            conn.close()
            return True
            
        except sqlite3.DatabaseError as e:
            print(f"❌ SQLite database error: {e}")
            return False
        except Exception as e:
            print(f"❌ SQLite access failed: {e}")
            return False
            
    except ImportError:
        print("❌ sqlite3 module not available")
        return False

def test_ase_access(db_path):
    """Test ASE database access."""
    print("\n⚛️  ASE ACCESS TEST")
    print("-" * 40)
    
    try:
        from ase.db import connect
        
        # Try to connect with ASE
        try:
            db = connect(db_path)
            print("✅ ASE database connection successful")
            
            # Try to count rows without loading all data
            try:
                count = 0
                for i, row in enumerate(db.select()):
                    count += 1
                    if i >= 10:  # Limit to first 10 rows for testing
                        break
                        
                print(f"✅ Successfully accessed {count} molecules (tested first 10)")
                return True, count
                
            except Exception as e:
                print(f"❌ Error iterating through database: {e}")
                
                # Try alternative counting method
                try:
                    rows = list(db.select())
                    count = len(rows)
                    print(f"⚠️  Alternative access method found {count} molecules")
                    return True, count
                except Exception as e2:
                    print(f"❌ Alternative access also failed: {e2}")
                    return False, 0
                    
        except Exception as e:
            print(f"❌ ASE database connection failed: {e}")
            return False, 0
            
    except ImportError:
        print("❌ ASE module not available")
        return False, 0

def test_molecule_access(db_path):
    """Test accessing individual molecules."""
    print("\n🧬 MOLECULE ACCESS TEST")
    print("-" * 40)
    
    try:
        from ase.db import connect
        
        db = connect(db_path)
        molecule_count = 0
        successful_reads = 0
        errors = []
        
        for i, row in enumerate(db.select()):
            molecule_count += 1
            try:
                atoms = row.toatoms()
                symbols = atoms.get_chemical_symbols()
                positions = atoms.positions
                successful_reads += 1
                
                if i == 0:  # Print details for first molecule
                    print(f"✅ First molecule: {len(symbols)} atoms, elements: {set(symbols)}")
                    
            except Exception as e:
                error_msg = str(e)
                errors.append(f"Molecule {i+1}: {error_msg}")
                
            if i >= 100:  # Limit testing to first 100 molecules
                break
                
        print(f"📊 Processed {molecule_count} molecules")
        print(f"✅ Successfully read {successful_reads} molecules")
        
        if errors:
            print(f"❌ {len(errors)} molecules had errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")
        
        return successful_reads > 0
        
    except Exception as e:
        print(f"❌ Molecule access test failed: {e}")
        return False

def test_edm_compatibility(db_path):
    """Test compatibility with E3 Diffusion system."""
    print("\n🔬 E3 DIFFUSION COMPATIBILITY TEST")
    print("-" * 40)
    
    try:
        # Test if our molecular_db_utils can handle this database
        import sys
        import os
        
        # Add current directory to path to import our modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from qm9.general_molecular_db import analyze_ase_database, validate_molecular_database
        
        # Test validation
        try:
            is_valid, issues, recommendations = validate_molecular_database(db_path)
            print(f"📋 Validation result: {'✅ VALID' if is_valid else '⚠️  HAS ISSUES'}")
            
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"   - {issue}")
            
            if recommendations:
                print("Recommendations:")
                for rec in recommendations[:3]:  # Show first 3 recommendations
                    print(f"   - {rec}")
                if len(recommendations) > 3:
                    print(f"   ... and {len(recommendations) - 3} more recommendations")
                    
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            
        # Test analysis
        try:
            analysis = analyze_ase_database(db_path)
            print(f"✅ Analysis successful:")
            print(f"   - {analysis['total_molecules']} molecules")
            print(f"   - {len(analysis['unique_elements'])} unique elements")
            print(f"   - Max {analysis['max_atoms']} atoms per molecule")
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            print(f"   Full error: {traceback.format_exc()}")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import E3 Diffusion modules: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Debug ASE database files for E3 Diffusion compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_ase_database.py molecules.db
  python debug_ase_database.py --verbose /path/to/database.db
        """
    )
    
    parser.add_argument('db_path', help='Path to the ASE database file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed error messages and stack traces')
    
    args = parser.parse_args()
    
    print("🔧 ASE DATABASE DEBUG TOOL")
    print("=" * 60)
    print(f"Database: {args.db_path}")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Basic File Access", test_basic_file_access),
        ("SQLite Access", test_sqlite_access),
        ("ASE Access", test_ase_access),
        ("Molecule Access", test_molecule_access),
        ("E3 Diffusion Compatibility", test_edm_compatibility)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "ASE Access":
                success, count = test_func(args.db_path)
                results[test_name] = success
            else:
                success = test_func(args.db_path)
                results[test_name] = success
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            if args.verbose:
                print(f"Full traceback:\n{traceback.format_exc()}")
            results[test_name] = False
    
    # Summary
    print("\n📋 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    # Overall assessment
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Database appears to be fully functional!")
    elif passed_tests >= total_tests - 1:
        print("✅ Database is mostly functional with minor issues")
    elif passed_tests >= 2:
        print("⚠️  Database has significant issues but may be partially usable")
    else:
        print("❌ Database appears to be severely corrupted or incompatible")
    
    print("\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    if not results.get("Basic File Access", False):
        print("- Check file path and permissions")
    elif not results.get("SQLite Access", False):
        print("- File may not be a valid SQLite database")
        print("- Try recreating the database from source data")
    elif not results.get("ASE Access", False):
        print("- Database may be corrupted or incompatible with current ASE version")
        print("- Try updating ASE: pip install --upgrade ase")
    elif not results.get("Molecule Access", False):
        print("- Some molecules in the database may be corrupted")
        print("- Consider filtering or regenerating problematic entries")
    elif not results.get("E3 Diffusion Compatibility", False):
        print("- Database structure may be incompatible with E3 Diffusion")
        print("- Check if all required properties are present")
    else:
        print("- Database appears healthy!")
        print("- If you're still having issues, check your E3 Diffusion installation")

if __name__ == "__main__":
    main()