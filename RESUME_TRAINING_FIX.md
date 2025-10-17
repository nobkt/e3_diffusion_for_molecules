# Resume Training Fix - Summary

## Problem

When running `example_resume_training.sh` to resume training, the following error occurred:

```
RuntimeError: Error(s) in loading state_dict for EnVariationalDiffusion:
	size mismatch for dynamics.egnn.embedding.weight: copying a param with shape torch.Size([256, 39]) from checkpoint, the shape in current model is torch.Size([256, 33]).
```

## Root Cause

When resuming training, the following sequence caused the issue:

1. Load saved training parameters from `args.pickle`
2. However, `context_node_nf` is then **recalculated** from the dataset
3. The recalculated value may differ from the original training
4. Create model with new `context_node_nf`
5. But checkpoint was saved with old `context_node_nf`
6. Result: Embedding layer size mismatch error (39 vs 33 features)

Specifically:
- Checkpoint: 11 atom types + 1 time + 27 context features = 39 features
- Recreated model: 11 atom types + 1 time + 21 context features = 33 features (different!)

## Solution

Modified `main_qm9.py` to use the saved `context_node_nf` when resuming training:

### Implementation

```python
# When resuming, preserve context_node_nf from saved args
if args.resume is not None and hasattr(args, 'context_node_nf'):
    # Use the saved context_node_nf from the checkpoint
    context_node_nf = args.context_node_nf
    print(f'Resuming training: using saved context_node_nf = {context_node_nf}')
    
    # Still compute property_norms for conditioning
    if len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
else:
    # Normal training: calculate context_node_nf from data
    if len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
```

### Key Points

1. **When resuming**: If `args.resume` is set and `args.context_node_nf` exists
   - Use the saved `context_node_nf` directly
   - Ensures model architecture matches the checkpoint

2. **When starting new training**: If `args.resume` is not set
   - Calculate `context_node_nf` from data (as before)
   - No change in behavior

3. **Backward compatibility**: For old checkpoints (without saved `context_node_nf`)
   - `hasattr` check provides fallback
   - Recalculates from data

## Test Results

All tests passed successfully:

```
✓ context_node_nf preservation: PASSED
✓ context_node_nf calculation for new training: PASSED
✓ Backward compatibility: PASSED
✓ Bug scenario reproduction: PASSED
✓ All existing resume tests: PASSED
✓ Integration test: PASSED
  - WITHOUT fix: 39 features vs 33 expected → SIZE MISMATCH ✗
  - WITH fix: 39 features vs 39 expected → SUCCESS ✓
```

## Usage

After the fix, resume training works correctly:

```bash
# Resume from checkpoint directory
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --resume outputs/exp_cond_molecular_descriptors \
    --n_epochs 500 \
    --no_wandb

# Or use the example script
bash example_resume_training.sh
```

## Impact

- ✅ Fixes the resume training error
- ✅ Maintains backward compatibility with old checkpoints
- ✅ No changes to new training behavior
- ✅ All existing tests still pass

## Technical Details

### Model Architecture

```
EGNN embedding layer input features = dynamics_in_node_nf + context_node_nf

where:
- in_node_nf = len(atom_decoder)  # Number of atom types (e.g., 11)
- dynamics_in_node_nf = in_node_nf + 1 (if condition_time)  # Including time conditioning
- context_node_nf = output dimension of prepare_context()  # Conditioning features

Example:
- in_node_nf = 11 (H, C, N, O, F, Si, P, S, Cl, Br, I)
- dynamics_in_node_nf = 12 (11 + 1 for time)
- context_node_nf = 27 (molecular_weight + pi_conjugation_ratio + atom_types_encoding + functional_groups_encoding)
- total = 12 + 27 = 39
```

### Why the Error Occurred

1. During training: Dataset A → `context_node_nf = 27`, save model
2. During resume: Dataset B → recalculate `context_node_nf = 21` (different value)
3. Create new model: `total = 12 + 21 = 33` features
4. Load checkpoint: expects `39` features
5. Error: `33 != 39`

### How the Fix Works

By using saved `context_node_nf = 27` during resume:
- New model: `total = 12 + 27 = 39` features
- Checkpoint: `39` features
- Success: `39 == 39` ✓

## Related Files

- `main_qm9.py`: Main fix (lines 267-290)
- `test_resume_context_nf.py`: Tests for context_node_nf preservation
- `test_resume_integration.py`: Integration test (reproduces actual bug scenario)
- `test_resume.py`: Existing resume training tests
- `RESUME_TRAINING_FIX_JA.md`: Japanese version of this document

## Additional Information

This fix is important for ensuring model architecture consistency, especially in scenarios such as:

1. Resuming training with a different dataset
2. When conditioning settings have changed
3. When dataset statistics have changed

The fix ensures that when resuming from a checkpoint, the saved architecture is always used, preventing errors.
