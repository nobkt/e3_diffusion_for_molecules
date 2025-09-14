#!/bin/bash

# E3 Diffusion for Molecules - Quick Installation Script
# This script automates the installation process for GPU environments

set -e  # Exit on any error

echo "🚀 E3 Diffusion for Molecules - Installation Script"
echo "=================================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    USE_CONDA=true
    echo "✓ Conda detected"
else
    USE_CONDA=false
    echo "⚠ Conda not detected, using pip"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📍 Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.8" ]] || [[ "$PYTHON_VERSION" > "3.11" ]]; then
    echo "❌ Python version $PYTHON_VERSION is not supported. Please use Python 3.8-3.11"
    exit 1
fi

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "📍 CUDA Version: $CUDA_VERSION"
    USE_GPU=true
else
    echo "⚠ No NVIDIA GPU detected, installing CPU version"
    USE_GPU=false
fi

# Create environment
if [ "$USE_CONDA" = true ]; then
    echo "🔧 Creating conda environment..."
    conda create -n edm python=3.10 -y || true
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate edm
    
    # Install RDKit via conda
    echo "📦 Installing RDKit via conda..."
    conda install -c conda-forge rdkit -y
else
    echo "🔧 Creating virtual environment..."
    python3 -m venv edm_env
    source edm_env/bin/activate
fi

# Install PyTorch with appropriate CUDA support
echo "🔥 Installing PyTorch..."
if [ "$USE_GPU" = true ]; then
    if [[ "$CUDA_VERSION" == "11.7" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    elif [[ "$CUDA_VERSION" == "11.8" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" >= "12.1" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.5.0
pip install ase>=3.22.0 wandb>=0.12.0 tqdm>=4.62.0 imageio>=2.9.0
pip install seaborn>=0.11.0 networkx>=2.6.0 msgpack>=1.0.0 scikit-learn>=1.0.0

# Install RDKit if not using conda
if [ "$USE_CONDA" = false ]; then
    echo "📦 Installing RDKit via pip..."
    pip install rdkit || echo "⚠ RDKit installation failed, some features may be limited"
fi

# Install OpenBabel
echo "📦 Installing OpenBabel..."
if [ "$USE_CONDA" = true ]; then
    conda install -c conda-forge openbabel -y || pip install openbabel-wheel
else
    pip install openbabel-wheel
fi

# Install the package in development mode
echo "📦 Installing package..."
pip install -e .

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import numpy as np
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU device: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

try:
    from rdkit import Chem
    print('✓ RDKit available')
except ImportError:
    print('⚠ RDKit not available')

try:
    from qm9.models import get_model
    print('✓ E3 Diffusion models can be imported')
except ImportError as e:
    print(f'❌ Model import failed: {e}')
"

echo ""
echo "🎉 Installation completed!"
echo ""
echo "Next steps:"
if [ "$USE_CONDA" = true ]; then
    echo "1. Activate environment: conda activate edm"
else
    echo "1. Activate environment: source edm_env/bin/activate"
fi
echo "2. Configure wandb: wandb login"
echo "3. Test training: python main_qm9.py --exp_name test --n_epochs 1 --batch_size 32"
echo ""
echo "For detailed usage, see INSTALLATION_GUIDE.md"