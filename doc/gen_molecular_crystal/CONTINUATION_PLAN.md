# Property-Conditioned Crystal Generation - Continuation Plan

**Document Type**: Implementation Plan & Detailed Specification  
**Phase**: 3 (Training & Generation Pipeline)  
**Status**: Planning  
**Date**: 2025-10-24  
**Version**: 1.0

---

## Overview

This document provides a detailed implementation plan, specification, and design for **Phase 3** of the property-conditioned molecular crystal generation system. Phase 1 (Core Modules) and Phase 2 (Dataset Extension) have been completed successfully.

---

## Phase 3: Training & Generation Pipeline

### Goals

1. Provide command-line tools for training and generation with property conditioning
2. Enable end-to-end workflow from data preparation to crystal generation
3. Integrate property conditioning into existing training infrastructure
4. Validate generated crystals against target properties

---

## Component P3-1: Training Script

### File: `main_crystal_with_properties.py`

### Purpose
Command-line interface for training crystal generation models with property conditioning.

### Requirements

**Functional Requirements**:
- FR-P3-1.1: Accept property names as command-line arguments
- FR-P3-1.2: Load datasets with properties using CrystalDatasetWithProperties
- FR-P3-1.3: Initialize ExtendedCombinedConditioning with property module
- FR-P3-1.4: Save property statistics with model checkpoint
- FR-P3-1.5: Support resume training with property conditioning
- FR-P3-1.6: Log property-specific metrics during training

**Non-Functional Requirements**:
- NFR-P3-1.1: Training time < 5% overhead compared to non-property training
- NFR-P3-1.2: Memory usage < 10% increase
- NFR-P3-1.3: Backward compatible with existing training scripts

### Interface Specification

**Command-Line Arguments**:
```bash
python main_crystal_with_properties.py \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --property_names bandgap melting_point dielectric_constant \
    --conditioning molecular space_group density property \
    --conditioning_dim 256 \
    --property_hidden_dim 512 \
    --property_n_layers 3 \
    --n_epochs 500 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --exp_name crystal_with_properties \
    --output_dir outputs/
```

**New Arguments**:
- `--property_names`: List of property names to condition on
- `--property_hidden_dim`: Hidden dimension for property MLP (default: 512)
- `--property_n_layers`: Number of layers in property MLP (default: 3)

### Implementation Pseudocode

```python
def main():
    # 1. Parse arguments
    args = parse_args()
    
    # 2. Setup logging and directories
    setup_logging(args.exp_name)
    create_output_dirs(args.output_dir)
    
    # 3. Load datasets
    train_dataset = CrystalDatasetWithProperties(
        db_path=args.crystal_db_path,
        indices=train_indices,
        property_names=args.property_names
    )
    val_dataset = CrystalDatasetWithProperties(
        db_path=args.crystal_db_path,
        indices=val_indices,
        property_names=args.property_names
    )
    
    # 4. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_crystal_batch
    )
    
    # 5. Create conditioning modules
    property_conditioning = None
    if 'property' in args.conditioning:
        property_conditioning = PropertyConditioning(
            property_names=args.property_names,
            conditioning_dim=args.conditioning_dim,
            hidden_dim=args.property_hidden_dim,
            n_layers=args.property_n_layers
        )
        property_conditioning.set_normalization_params(
            train_dataset.property_mean,
            train_dataset.property_std
        )
    
    combined_conditioning = ExtendedCombinedConditioning(
        molecular_conditioning=molecular_cond,
        space_group_embedding=sg_embed if 'space_group' in args.conditioning else None,
        density_conditioning=dens_cond if 'density' in args.conditioning else None,
        property_conditioning=property_conditioning,
        conditioning_dim=args.conditioning_dim
    )
    
    # 6. Create diffusion model
    model = CrystalDiffusionModel(
        conditioning_module=combined_conditioning,
        ...
    ).to(args.device)
    
    # 7. Training loop
    for epoch in range(args.n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate_epoch(model, val_loader)
        
        # Save checkpoint including property statistics
        if epoch % args.save_interval == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'property_mean': train_dataset.property_mean,
                'property_std': train_dataset.property_std,
                'property_names': args.property_names,
                'args': args
            }, f'{args.output_dir}/checkpoint_epoch_{epoch}.pt')
        
        log_metrics(epoch, train_loss, val_loss)

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Get conditioning
        conditioning = model.conditioning_module(
            batch['molecular_features'],
            space_group=batch.get('space_group'),
            density=batch.get('density'),
            properties=batch.get('properties')
        )
        
        # Compute diffusion loss
        loss = model.compute_loss(
            batch['positions'],
            batch['cell'],
            conditioning=conditioning
        )
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Testing Requirements

**Unit Tests** (`tests/test_training_with_properties.py`):
1. Test argument parsing
2. Test dataset creation with properties
3. Test conditioning module initialization
4. Test checkpoint save/load with property statistics
5. Test training loop (1 epoch, small dataset)

**Integration Tests**:
1. End-to-end training (10 epochs, synthetic data)
2. Resume training from checkpoint
3. Validate property statistics in checkpoint
4. Memory profiling test

---

## Component P3-2: Generation Script

### File: `generate_crystal_with_all_conditions.py`

### Purpose
Command-line interface for generating crystals with target property values.

### Requirements

**Functional Requirements**:
- FR-P3-2.1: Accept target property values as arguments
- FR-P3-2.2: Load model checkpoint with property statistics
- FR-P3-2.3: Generate crystals conditioned on target properties
- FR-P3-2.4: Save generated crystals in CIF format
- FR-P3-2.5: Output metadata including target properties
- FR-P3-2.6: Support batch generation (multiple property targets)

### Interface Specification

**Command-Line Arguments**:
```bash
python generate_crystal_with_all_conditions.py \
    --model_path outputs/crystal_with_properties/best_model.pt \
    --molecule_db_path data/molecules.db \
    --molecule_id benzene_001 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --target_dielectric_constant 3.0 \
    --space_group 14 \
    --density 1.2 \
    --n_samples 100 \
    --output_dir generated_crystals/ \
    --output_format cif
```

**New Arguments**:
- `--target_*`: Target property values (dynamic based on property names)
- `--property_file`: Optional CSV file with multiple target property sets

### Implementation Pseudocode

```python
def main():
    # 1. Parse arguments
    args = parse_args()
    
    # 2. Load checkpoint
    checkpoint = torch.load(args.model_path)
    property_names = checkpoint['property_names']
    property_mean = checkpoint['property_mean']
    property_std = checkpoint['property_std']
    
    # 3. Reconstruct model
    model = load_model_from_checkpoint(checkpoint)
    model.eval()
    
    # 4. Load molecule data
    molecule_data = load_molecule(args.molecule_db_path, args.molecule_id)
    molecular_features = extract_molecular_features(molecule_data)
    
    # 5. Prepare target properties
    target_properties = []
    for prop_name in property_names:
        value = getattr(args, f'target_{prop_name}', None)
        if value is None:
            raise ValueError(f"Missing target value for property: {prop_name}")
        target_properties.append(value)
    target_properties = torch.tensor([target_properties])
    
    # 6. Get conditioning
    conditioning = model.conditioning_module(
        molecular_features,
        space_group=args.space_group,
        density=args.density,
        properties=target_properties
    )
    
    # 7. Generate crystals
    generated_crystals = []
    for i in range(args.n_samples):
        crystal = model.sample(
            conditioning=conditioning,
            num_atoms=molecule_data['num_atoms'],
            device=args.device
        )
        generated_crystals.append(crystal)
    
    # 8. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    for i, crystal in enumerate(generated_crystals):
        # Save CIF
        cif_path = f'{args.output_dir}/crystal_{i:04d}.cif'
        save_cif(crystal, cif_path)
        
        # Save metadata
        metadata = {
            'crystal_id': f'crystal_{i:04d}',
            'molecule_id': args.molecule_id,
            'target_properties': {
                name: value 
                for name, value in zip(property_names, target_properties[0].tolist())
            },
            'space_group': args.space_group,
            'density': args.density,
            'generation_time': datetime.now().isoformat()
        }
        metadata_path = f'{args.output_dir}/crystal_{i:04d}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # 9. Generate summary
    summary = {
        'n_samples': args.n_samples,
        'molecule_id': args.molecule_id,
        'target_properties': {
            name: value 
            for name, value in zip(property_names, target_properties[0].tolist())
        },
        'output_dir': args.output_dir
    }
    with open(f'{args.output_dir}/generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {args.n_samples} crystals in {args.output_dir}")
```

### Testing Requirements

**Unit Tests** (`tests/test_generation_with_properties.py`):
1. Test argument parsing
2. Test checkpoint loading
3. Test property target preparation
4. Test conditioning generation
5. Test CIF output
6. Test metadata generation

**Integration Tests**:
1. End-to-end generation (trained model + target properties)
2. Batch generation with multiple targets
3. Validate output file structure

---

## Component P3-3: Data Preparation Script

### File: `scripts/prepare_property_dataset.py`

### Purpose
Prepare and validate crystal datasets with property values.

### Requirements

**Functional Requirements**:
- FR-P3-3.1: Merge property CSV with crystal database
- FR-P3-3.2: Validate data completeness
- FR-P3-3.3: Check for zero-variance properties
- FR-P3-3.4: Compute and display statistics
- FR-P3-3.5: Create train/val/test splits

### Interface Specification

```bash
python scripts/prepare_property_dataset.py \
    --crystal_db data/crystals.db \
    --property_csv data/crystal_properties.csv \
    --output_db data/crystals_with_props.db \
    --property_names bandgap melting_point dielectric_constant \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --validate
```

### Implementation Pseudocode

```python
def main():
    # 1. Parse arguments
    args = parse_args()
    
    # 2. Load crystal database
    crystal_db = connect(args.crystal_db)
    
    # 3. Load property CSV
    properties_df = pd.read_csv(args.property_csv)
    
    # 4. Validate data
    if args.validate:
        validate_properties(crystal_db, properties_df, args.property_names)
    
    # 5. Merge properties into database
    output_db = connect(args.output_db)
    for row in crystal_db.select():
        crystal_id = row.data.get('crystal_id', row.id)
        
        # Get properties for this crystal
        prop_row = properties_df[properties_df['crystal_id'] == crystal_id]
        if len(prop_row) == 0:
            raise ValueError(f"No properties found for crystal {crystal_id}")
        
        # Update data with properties
        data = row.data.copy()
        for prop_name in args.property_names:
            if prop_name not in prop_row.columns:
                raise ValueError(f"Property {prop_name} not in CSV")
            data[prop_name] = prop_row[prop_name].values[0]
        
        # Write to output database
        atoms = row.toatoms()
        output_db.write(atoms, data=data)
    
    # 6. Compute and display statistics
    compute_and_display_statistics(output_db, args.property_names)
    
    # 7. Create splits
    create_splits(
        output_db,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=Path(args.output_db).parent
    )

def validate_properties(db, properties_df, property_names):
    """Validate property data completeness and variance."""
    # Check all crystals have properties
    crystal_ids_db = set(row.data.get('crystal_id', row.id) for row in db.select())
    crystal_ids_csv = set(properties_df['crystal_id'])
    
    missing_in_csv = crystal_ids_db - crystal_ids_csv
    if missing_in_csv:
        raise ValueError(f"Properties missing for crystals: {missing_in_csv}")
    
    # Check property variance
    for prop_name in property_names:
        values = properties_df[prop_name].values
        if np.std(values) < 1e-8:
            raise ValueError(f"Property {prop_name} has zero variance")
    
    print("✓ All validation checks passed")
```

---

## Component P3-4: Integration Tests

### File: `tests/test_integration_property_conditioning.py`

### Test Cases

**IT-001: End-to-End Training**
```python
def test_end_to_end_training():
    """Test complete training workflow with properties."""
    # 1. Create synthetic dataset
    create_synthetic_crystal_db_with_properties(
        'test_crystals.db',
        n_crystals=100,
        property_names=['bandgap', 'melting_point']
    )
    
    # 2. Run training (5 epochs)
    run_training_script(
        crystal_db='test_crystals.db',
        property_names=['bandgap', 'melting_point'],
        n_epochs=5
    )
    
    # 3. Validate checkpoint
    checkpoint = torch.load('outputs/test/checkpoint_epoch_5.pt')
    assert 'property_mean' in checkpoint
    assert 'property_std' in checkpoint
    assert 'property_names' in checkpoint
```

**IT-002: End-to-End Generation**
```python
def test_end_to_end_generation():
    """Test complete generation workflow with target properties."""
    # 1. Load trained model
    model = load_trained_model('test_model.pt')
    
    # 2. Generate with target properties
    crystals = generate_crystals_with_properties(
        model=model,
        molecule_id='test_mol',
        target_properties={'bandgap': 2.5, 'melting_point': 180.0},
        n_samples=10
    )
    
    # 3. Validate outputs
    assert len(crystals) == 10
    for crystal in crystals:
        assert validate_crystal_structure(crystal)
```

---

## Implementation Timeline

### Week 1: Training Script
- Day 1-2: Argument parsing and dataset loading
- Day 3-4: Model initialization with property conditioning
- Day 5-6: Training loop and checkpoint management
- Day 7: Testing and debugging

### Week 2: Generation Script
- Day 1-2: Checkpoint loading and model reconstruction
- Day 3-4: Generation with property targets
- Day 5-6: Output formatting (CIF, metadata)
- Day 7: Testing and debugging

### Week 3: Data Preparation & Integration Tests
- Day 1-2: Data preparation script
- Day 3-4: Integration tests
- Day 5-6: Documentation and examples
- Day 7: Final validation and review

---

## Success Criteria

### Phase 3 Completion Criteria

1. **Functionality**:
   - ✅ Training script supports property conditioning
   - ✅ Generation script accepts target properties
   - ✅ Data preparation validates and merges data
   - ✅ All integration tests pass

2. **Performance**:
   - ✅ Training overhead < 5%
   - ✅ Memory overhead < 10%
   - ✅ Generation time comparable to baseline

3. **Quality**:
   - ✅ Code coverage > 80%
   - ✅ All linters pass
   - ✅ Documentation complete

4. **Validation**:
   - ✅ End-to-end workflow demonstrated
   - ✅ Generated crystals are physically valid
   - ✅ Property MAE < 10% (if validation model available)

---

## Risk Assessment

### Risk R-1: Property Prediction Validation

**Description**: Without ground-truth property calculator, cannot validate if generated crystals actually have target properties.

**Mitigation**:
1. Train separate property prediction model
2. Use existing DFT calculators (if available)
3. Validate structural plausibility instead
4. Document limitation in user guide

**Priority**: Medium

### Risk R-2: Training Stability

**Description**: Property conditioning may affect training stability.

**Mitigation**:
1. Careful hyperparameter tuning
2. Gradient clipping
3. Monitor for NaN losses
4. Start with single property, add incrementally

**Priority**: High

### Risk R-3: Data Quality

**Description**: Property data quality varies across sources.

**Mitigation**:
1. Validation script catches obvious errors
2. Outlier detection
3. Manual review of statistics
4. Document data requirements clearly

**Priority**: Medium

---

## Dependencies

### External Dependencies
- PyTorch >= 1.12.0
- ASE >= 3.22.0
- pandas (for CSV handling)
- numpy, scipy

### Internal Dependencies
- Phase 1: PropertyConditioning, ExtendedCombinedConditioning
- Phase 2: CrystalDatasetWithProperties
- Existing: CrystalDiffusionModel, training infrastructure

---

## Documentation Requirements

### User Documentation
1. Training guide with property conditioning
2. Generation guide with target properties
3. Data preparation guide
4. Examples and tutorials
5. Troubleshooting guide

### Developer Documentation
1. API documentation for new functions
2. Architecture diagram updates
3. Training loop diagram with properties
4. Testing guide

---

## Conclusion

This continuation plan provides a complete specification for Phase 3 of the property-conditioned crystal generation system. The implementation is designed to integrate seamlessly with the completed Phase 1 and 2 components while maintaining the core principles of no fallbacks, E(3) equivariance, and modular design.

Estimated completion time: 3 weeks (1 developer)  
Estimated lines of code: ~1500 (implementation + tests)  
Estimated test coverage: >80%

---

**Document Status**: Ready for Implementation  
**Approved By**: [Pending Review]  
**Last Updated**: 2025-10-24
