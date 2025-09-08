#!/usr/bin/env python3
"""
Test script to demonstrate the fix for database validation issues.

This script shows the difference between the old behavior (crash) and 
the new behavior (graceful handling) of malformed ASE databases.
"""

import os
import sys
import tempfile
from ase.atoms import Atoms
from ase.db import connect

def create_test_databases():
    """Create test databases for demonstration."""
    # Create a good database
    good_db = 'demo_good.db'
    if os.path.exists(good_db):
        os.remove(good_db)
    
    db = connect(good_db)
    molecules = [
        Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]]),
        Atoms('CH4', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
    ]
    
    for i, mol in enumerate(molecules):
        db.write(mol, data={'energy': -10.0 - i})
    
    print(f"✅ Created good database: {good_db}")
    
    # Create a malformed database by truncating
    malformed_db = 'demo_malformed.db'
    if os.path.exists(malformed_db):
        os.remove(malformed_db)
    
    # Copy and truncate to create corruption
    import shutil
    shutil.copy(good_db, malformed_db)
    
    # Truncate the file to make it malformed
    with open(malformed_db, 'r+b') as f:
        f.truncate(1024)  # Keep only first 1KB
    
    print(f"⚠️ Created malformed database: {malformed_db}")
    
    return good_db, malformed_db

def test_database_analysis():
    """Test database analysis with both good and malformed databases."""
    print("\n" + "="*60)
    print("DEMONSTRATION: ASE Database Malformed Error Fix")
    print("="*60)
    
    good_db, malformed_db = create_test_databases()
    
    # Test with good database
    print(f"\n📊 Testing analysis with GOOD database ({good_db}):")
    print("-" * 50)
    os.system(f"python molecular_db_utils.py analyze --db_path {good_db}")
    
    # Test with malformed database 
    print(f"\n💥 Testing analysis with MALFORMED database ({malformed_db}):")
    print("-" * 50)
    print("Before fix: This would crash with 'DATABASE VALIDATION FAILED'")
    print("After fix: Graceful handling with warnings...")
    print()
    os.system(f"python molecular_db_utils.py analyze --db_path {malformed_db}")
    
    # Test debug script
    print(f"\n🔧 Debug script analysis of MALFORMED database:")
    print("-" * 50)
    os.system(f"python debug_ase_database.py {malformed_db}")
    
    # Cleanup
    for db in [good_db, malformed_db]:
        if os.path.exists(db):
            os.remove(db)
    print(f"\n🧹 Cleaned up test databases")

if __name__ == "__main__":
    test_database_analysis()