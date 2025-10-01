# Quick Fix Guide - CSV Export Bug

## What Was Fixed?
The CSV export for generation conditions had bugs where composition strings didn't match the encodings in `atom_types_encoding.csv` and `functional_groups_encoding.csv`.

## How to Verify the Fix?

### 1. Run Tests
```bash
# Quick test
python validate_fix.py

# Comprehensive test
python test_all_conditions.py

# Unit tests
python test_csv_export_fix.py
```

All should show ✅ PASS.

### 2. Export CSV with Your Database
```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path your_database.db \
    --export_conditions_csv ./output \
    --no_wandb \
    --remove_h
```

### 3. Check the Output
Open `output/atom_types_encoding.csv`:
```
ID,分子の組成,C,Cl,N,O,P,S
0,C5Cl2PS,1,1,0,0,1,1
```

Verify:
- ✅ Composition `C5Cl2PS` contains C, Cl, P, S
- ✅ Encoding shows C=1, Cl=1, P=1, S=1
- ✅ They match!

## Training with Conditions

Now you can train with generation conditions:

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path ase.db \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
    --exp_name my_conditional_model
```

All four conditions will work correctly!

## Need Help?

See detailed documentation:
- `CSV_EXPORT_BUG_FIX.md` - Technical details
- `FIX_SUMMARY.md` - Quick summary
