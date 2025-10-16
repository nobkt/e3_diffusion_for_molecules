# Resume Training Fix - Summary

## Problem
User wanted to resume training from epoch 200 but the existing resume functionality was broken:
- Models are saved as `generative_model.npy` and `generative_model_ema.npy`
- But the code tried to load from `flow.npy` (which doesn't exist)

## Solution
Fixed the resume functionality in `main_qm9.py` to:
1. Load from the correct model file names
2. Support both directory and file paths
3. Automatically detect the epoch to resume from
4. Maintain backward compatibility

## Files Changed

### 1. main_qm9.py (Core Fix)
- **Lines 173-223**: Enhanced args loading logic
  - Support directory or file path for `--resume`
  - Load args.pickle if available
  - Auto-detect start epoch from checkpoint
  
- **Lines 297-341**: Enhanced model loading logic
  - Try `generative_model_ema.npy` first (preferred)
  - Fall back to `generative_model.npy`
  - Fall back to `flow.npy` (backward compatibility)
  - Load optimizer state if available
  - Clear error messages if checkpoint not found

### 2. test_resume.py (New Test File)
- Tests resume from directory
- Tests resume from file path
- Tests backward compatibility with old format
- Tests with existing checkpoints
- All tests pass ✓

### 3. Documentation (Enhanced)
- **doc/user_manual.md**: Updated with detailed examples
- **RESUME_TRAINING.md**: Comprehensive guide (English)
- **RESUME_TRAINING_JA.md**: Japanese guide for the user
- **example_resume_training.sh**: Interactive helper script

## How to Use

### Simple Case (Recommended)
```bash
python main_qm9.py \
    --exp_name your_experiment \
    --resume outputs/your_experiment \
    --n_epochs 500
```

### For the Specific Problem Statement
```bash
# Resume the training that stopped at epoch 200
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --resume outputs/exp_cond_molecular_descriptors \
    --n_epochs 500 \
    --no_wandb
```

This will:
- ✓ Load model from `generative_model_ema.npy`
- ✓ Load optimizer from `optim.npy`
- ✓ Load config from `args.pickle`
- ✓ Resume from epoch 200 automatically
- ✓ Continue training to epoch 500
- ✓ Save to `outputs/exp_cond_molecular_descriptors_resume`

## Key Features

### 1. Automatic File Detection
```
Priority order:
1. generative_model_ema.npy (best quality, with EMA)
2. generative_model.npy (standard model)
3. flow.npy (backward compatibility)
```

### 2. Flexible Resume Options
```bash
# From directory
--resume outputs/my_experiment

# From specific file
--resume outputs/my_experiment/generative_model_ema.npy

# With explicit epoch
--resume outputs/my_experiment --start_epoch 200
```

### 3. Robust Error Handling
- Clear error if checkpoint not found
- Warning if optimizer state missing
- Graceful fallback for missing args.pickle

## Testing

```bash
# Run automated tests
python test_resume.py

# Run interactive example
./example_resume_training.sh
```

## Backward Compatibility

The fix maintains full backward compatibility:
- ✓ Works with old checkpoints using `flow.npy`
- ✓ Works with checkpoints without `current_epoch`
- ✓ Works with missing optimizer state
- ✓ Works with missing args.pickle

## Code Changes Summary

**Before:**
```python
# Old code (broken)
flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
model.load_state_dict(flow_state_dict)
optim.load_state_dict(optim_state_dict)
```

**After:**
```python
# New code (robust)
if os.path.isdir(args.resume):
    # Try multiple file names in priority order
    if os.path.exists(join(resume_dir, 'generative_model_ema.npy')):
        model_path = join(resume_dir, 'generative_model_ema.npy')
    elif os.path.exists(join(resume_dir, 'generative_model.npy')):
        model_path = join(resume_dir, 'generative_model.npy')
    elif os.path.exists(join(resume_dir, 'flow.npy')):
        model_path = join(resume_dir, 'flow.npy')
    else:
        raise FileNotFoundError(...)

flow_state_dict = torch.load(model_path)
model.load_state_dict(flow_state_dict)

# Gracefully handle missing optimizer
if os.path.exists(optim_path):
    optim_state_dict = torch.load(optim_path)
    optim.load_state_dict(optim_state_dict)
else:
    print("Warning: Starting with fresh optimizer")
```

## Documentation

- 📖 **English Guide**: `RESUME_TRAINING.md`
- 📖 **Japanese Guide**: `RESUME_TRAINING_JA.md`
- 📖 **User Manual**: `doc/user_manual.md` (section "Resuming Training")
- 🔧 **Helper Script**: `example_resume_training.sh`
- ✅ **Test Script**: `test_resume.py`

## Impact

This fix enables users to:
1. ✅ Resume training from checkpoints (was broken before)
2. ✅ Continue training for more epochs easily
3. ✅ Use both directory and file paths for resume
4. ✅ Benefit from automatic epoch detection
5. ✅ Work with any checkpoint format (old or new)

## Verification

All changes have been tested:
- ✅ Unit tests pass (test_resume.py)
- ✅ Syntax check passes
- ✅ Backward compatibility verified
- ✅ Works with existing checkpoints

## User Answer

To the user's question in Japanese:

> main_qm9.pyでaseのdbに対して下記コマンドで条件付き学習を200エポック行いましたが、
> 学習がまだ不十分だったため、途中結果からrestartして継続して訓練を行いたいですが、
> その方法を教えてください。

**Answer:**

```bash
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --resume outputs/exp_cond_molecular_descriptors \
    --n_epochs 500 \
    --no_wandb
```

このコマンドで：
- エポック200から自動的に再開されます
- エポック500まで学習が継続されます
- すべての設定が自動的に読み込まれます

詳細は `RESUME_TRAINING_JA.md` をご覧ください。
