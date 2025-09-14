# Complete Installation Guide for E3 Diffusion for Molecules (GPU Support)

This guide provides step-by-step instructions to set up the E3 Diffusion for Molecules project with full functionality and GPU support.

## Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- **Operating System**: Linux (Ubuntu 18.04+) or macOS
- **Python**: 3.8-3.11 (3.9 or 3.10 recommended)
- **CUDA**: 11.7 or later (for GPU support)

## 1. Environment Setup

### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n edm python=3.10 -y
conda activate edm

# Install conda-forge packages
conda install -c conda-forge rdkit -y
```

### Option B: Using virtualenv

```bash
# Create virtual environment
python -m venv edm_env
source edm_env/bin/activate  # On Windows: edm_env\Scripts\activate
```

## 2. Clone the Repository

```bash
git clone https://github.com/nobkt/e3_diffusion_for_molecules.git
cd e3_diffusion_for_molecules
```

## 3. Install PyTorch with CUDA Support

Choose the appropriate command based on your CUDA version:

### For CUDA 11.7:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For CPU only (not recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 4. Install Core Dependencies

```bash
# Install scientific computing libraries
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.5.0

# Install molecular chemistry libraries
pip install ase>=3.22.0
pip install openbabel-wheel  # Alternative: conda install -c conda-forge openbabel

# Install machine learning and utilities
pip install wandb>=0.12.0
pip install tqdm>=4.62.0
pip install imageio>=2.9.0

# Install RDKit if not using conda
# If using pip (may have compatibility issues):
pip install rdkit
# OR if conda is available:
# conda install -c conda-forge rdkit
```

## 5. Install Additional Dependencies

```bash
# For GEOM dataset processing
pip install msgpack>=1.0.0

# For molecular analysis
pip install networkx>=2.6.0
pip install scikit-learn>=1.0.0

# For visualization
pip install seaborn>=0.11.0
```

## 6. Install the Package

```bash
# Install in development mode
pip install -e .
```

## 7. Verify Installation

### Check GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Test basic functionality:
```bash
python -c "
import torch
import numpy as np
from qm9 import dataset
from configs.datasets_config import get_dataset_info
print('✓ Basic imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU device: {torch.cuda.get_device_name(0)}')
"
```

### Test RDKit (if installed):
```bash
python -c "
try:
    from rdkit import Chem
    print('✓ RDKit imported successfully')
except ImportError:
    print('⚠ RDKit not available - some molecular analysis features may be limited')
"
```

## 8. Data Preparation

### For QM9 Dataset:
The QM9 dataset will be automatically downloaded when you first run training. No manual setup required.

### For GEOM-Drugs Dataset:
```bash
# Navigate to data directory
cd data/geom

# Download GEOM dataset (Warning: ~50GB)
wget https://dataverse.harvard.edu/api/access/datafile/4360331

# Extract the dataset
tar -xzvf 4360331

# Process the dataset
cd ../..
python build_geom_dataset.py
```

## 9. Configure Weights & Biases (Optional but Recommended)

```bash
# Login to wandb for experiment tracking
wandb login
```

If you don't have a wandb account, create one at https://wandb.ai/

## 10. Test Training

### Quick test with QM9:
```bash
python main_qm9.py \
    --exp_name test_installation \
    --n_epochs 1 \
    --batch_size 32 \
    --diffusion_steps 100 \
    --test_epochs 1
```

### Test with GPU (if available):
```bash
python main_qm9.py \
    --exp_name test_gpu \
    --n_epochs 1 \
    --batch_size 64 \
    --dp True \
    --diffusion_steps 100
```

## 11. Full Training Examples

### Train EDM on QM9:
```bash
python main_qm9.py \
    --n_epochs 3000 \
    --exp_name edm_qm9 \
    --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 9 \
    --lr 1e-4 \
    --normalize_factors [1,4,10] \
    --test_epochs 20 \
    --ema_decay 0.9999
```

### Train Conditional EDM:
```bash
python main_qm9.py \
    --exp_name exp_cond_alpha \
    --model egnn_dynamics \
    --lr 1e-4 \
    --nf 192 \
    --n_layers 9 \
    --save_model True \
    --diffusion_steps 1000 \
    --sin_embedding False \
    --n_epochs 3000 \
    --n_stability_samples 500 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --dequantization deterministic \
    --include_charges False \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --normalize_factors [1,8,1] \
    --conditioning alpha \
    --dataset qm9_second_half
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
   ```bash
   python main_qm9.py --batch_size 32  # or 16
   ```

2. **RDKit Import Error**: 
   ```bash
   conda install -c conda-forge rdkit
   ```

3. **OpenBabel Issues**:
   ```bash
   conda install -c conda-forge openbabel
   ```

4. **Wandb Login Issues**:
   ```bash
   wandb offline  # Run without wandb logging
   ```

5. **Memory Issues with GEOM Dataset**: The GEOM dataset is very large. Ensure you have sufficient disk space (>100GB) and RAM (>16GB).

### Performance Optimization:

1. **For Large GPUs** (>16GB VRAM):
   - Use larger batch sizes (128-256)
   - Increase model size: `--nf 512 --n_layers 12`

2. **For Smaller GPUs** (8-16GB VRAM):
   - Use smaller batch sizes (32-64)
   - Reduce model size: `--nf 128 --n_layers 6`

3. **For Multiple GPUs**:
   - Ensure `--dp True` is set
   - The code automatically detects and uses all available GPUs

### Verification Commands:

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Check all dependencies
python -c "
import numpy, scipy, matplotlib, torch, wandb, ase, tqdm, imageio
try: import rdkit; print('RDKit: OK')
except: print('RDKit: Not available')
try: import openbabel; print('OpenBabel: OK') 
except: print('OpenBabel: Not available')
print('All core dependencies: OK')
"

# Test model import
python -c "from qm9.models import get_model; print('Models import: OK')"
```

## Next Steps

After successful installation:

1. **Analysis**: Use `eval_analyze.py` to analyze sample quality
2. **Sampling**: Use `eval_sample.py` to generate new molecules
3. **Conditional Generation**: Train property-specific models
4. **Visualization**: Explore molecular visualizations

For detailed usage examples, see the main README.md and example scripts.

## Hardware Recommendations

### Minimum Requirements:
- GPU: NVIDIA GTX 1080 Ti (11GB VRAM)
- RAM: 16GB
- Storage: 200GB free space

### Recommended Configuration:
- GPU: NVIDIA RTX 3080/4080 or better (>16GB VRAM)
- RAM: 32GB or more
- Storage: 500GB+ SSD
- CPU: 8+ cores

### For Production/Research:
- GPU: NVIDIA A100, V100, or RTX 4090
- RAM: 64GB or more
- Storage: 1TB+ NVMe SSD
- CPU: 16+ cores