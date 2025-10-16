#!/bin/bash
# Example: Resume training from checkpoint
# 
# This script demonstrates how to resume training from a checkpoint
# using the exact scenario described in the problem statement.
#
# Original training command (stopped at epoch 200):
# The model was trained with conditional generation on molecular descriptors

echo "==========================================================="
echo "Example: Resume Training from Checkpoint"
echo "==========================================================="
echo ""

# Check if checkpoint exists
CHECKPOINT_DIR="outputs/exp_cond_molecular_descriptors"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "Make sure you have run the initial training command first:"
    echo ""
    echo "python main_qm9.py \\"
    echo "    --exp_name exp_cond_molecular_descriptors \\"
    echo "    --model egnn_dynamics \\"
    echo "    --lr 1e-4 \\"
    echo "    --nf 256 \\"
    echo "    --n_layers 9 \\"
    echo "    --save_model True \\"
    echo "    --diffusion_steps 1000 \\"
    echo "    --sin_embedding False \\"
    echo "    --n_epochs 200 \\"
    echo "    --n_stability_samples 1000 \\"
    echo "    --diffusion_noise_schedule polynomial_2 \\"
    echo "    --diffusion_noise_precision 1e-5 \\"
    echo "    --dequantization deterministic \\"
    echo "    --include_charges False \\"
    echo "    --diffusion_loss_type l2 \\"
    echo "    --batch_size 16 \\"
    echo "    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \\"
    echo "    --dataset ase_db \\"
    echo "    --ase_db_path ase.db \\"
    echo "    --test_epochs 10 \\"
    echo "    --no_wandb"
    echo ""
    exit 1
fi

echo "Found checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Check for required files
echo "Checking checkpoint files..."
if [ -f "$CHECKPOINT_DIR/generative_model_ema.npy" ]; then
    echo "✓ Found generative_model_ema.npy"
elif [ -f "$CHECKPOINT_DIR/generative_model.npy" ]; then
    echo "✓ Found generative_model.npy"
else
    echo "✗ Error: No model checkpoint found"
    exit 1
fi

if [ -f "$CHECKPOINT_DIR/optim.npy" ]; then
    echo "✓ Found optim.npy"
else
    echo "⚠ Warning: optim.npy not found (will use fresh optimizer)"
fi

if [ -f "$CHECKPOINT_DIR/args.pickle" ]; then
    echo "✓ Found args.pickle"
else
    echo "⚠ Warning: args.pickle not found (will use current arguments)"
fi

echo ""
echo "==========================================================="
echo "Resume Training Command"
echo "==========================================================="
echo ""
echo "To resume training for 300 more epochs (total 500 epochs), run:"
echo ""
echo "python main_qm9.py \\"
echo "    --exp_name exp_cond_molecular_descriptors \\"
echo "    --resume outputs/exp_cond_molecular_descriptors \\"
echo "    --n_epochs 500 \\"
echo "    --no_wandb"
echo ""
echo "Key points:"
echo "- The model, optimizer, and arguments will be loaded from the checkpoint"
echo "- Training will resume from the last saved epoch"
echo "- Results will be saved to: outputs/exp_cond_molecular_descriptors_resume"
echo "- You only need to specify the resume directory and new n_epochs"
echo ""
echo "==========================================================="
echo ""

# Optionally, ask user if they want to run it
read -p "Do you want to run the resume command now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    python main_qm9.py \
        --exp_name exp_cond_molecular_descriptors \
        --resume outputs/exp_cond_molecular_descriptors \
        --n_epochs 500 \
        --no_wandb
else
    echo "Skipped. You can run the command manually when ready."
fi
