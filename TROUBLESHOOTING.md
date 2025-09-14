# Troubleshooting Guide

This guide provides solutions to common issues encountered during installation and usage of E3 Diffusion for Molecules.

## Installation Issues

### 1. CUDA/GPU Issues

#### Problem: `RuntimeError: CUDA out of memory`
**Solution:**
```bash
# Reduce batch size
python main_qm9.py --batch_size 16  # or 8 for very limited memory

# Use gradient accumulation instead of large batches
python main_qm9.py --batch_size 16 --accumulate_grad_batches 4
```

#### Problem: `CUDA driver version is insufficient`
**Solution:**
- Update NVIDIA drivers to the latest version
- Install compatible CUDA toolkit
- Check compatibility: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

#### Problem: PyTorch not detecting GPU
**Solution:**
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Dependency Issues

#### Problem: `ModuleNotFoundError: No module named 'rdkit'`
**Solutions:**
```bash
# Option 1: Install via conda (recommended)
conda install -c conda-forge rdkit

# Option 2: Install via pip
pip install rdkit

# Option 3: If both fail, run without RDKit (limited functionality)
# The code should still work for basic training
```

#### Problem: `ModuleNotFoundError: No module named 'openbabel'`
**Solutions:**
```bash
# Option 1: Install via conda
conda install -c conda-forge openbabel

# Option 2: Install via pip
pip install openbabel-wheel

# Option 3: Install system package (Ubuntu/Debian)
sudo apt-get install openbabel python3-openbabel
```

#### Problem: `ImportError: cannot import name 'xyz' from 'module'`
**Solution:**
```bash
# Update all packages to latest versions
pip install --upgrade -r requirements.txt

# If still failing, try installing specific versions
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
```

### 3. Environment Issues

#### Problem: `Permission denied` errors
**Solution:**
```bash
# Fix permissions
chmod +x install.sh
chmod +x verify_installation.py

# Or run with sudo (not recommended for production)
sudo python main_qm9.py
```

#### Problem: Virtual environment issues
**Solution:**
```bash
# Completely remove and recreate environment
rm -rf edm_env  # or conda env remove -n edm
python -m venv edm_env
source edm_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Runtime Issues

### 1. Training Problems

#### Problem: `RuntimeError: Expected tensor to be on cuda:0 but got cpu`
**Solution:**
```bash
# Ensure all tensors are on GPU
python main_qm9.py --dp True  # Enable data parallel

# Or modify code to explicitly move tensors
# model = model.cuda()
# data = data.cuda()
```

#### Problem: Training is very slow
**Solutions:**
- Check GPU utilization: `nvidia-smi`
- Increase batch size if memory allows
- Use multiple GPUs: `--dp True`
- Reduce model complexity: `--nf 128 --n_layers 6`

#### Problem: `wandb.errors.AuthenticationError`
**Solutions:**
```bash
# Login to wandb
wandb login

# Or run offline
wandb offline

# Or disable wandb logging in code
# Comment out wandb.init() calls
```

### 2. Data Issues

#### Problem: QM9 dataset download fails
**Solution:**
```bash
# Manual download and setup
mkdir -p qm9/data
cd qm9/data
wget https://springernature.figshare.com/ndownloader/files/3195389
# The dataset will be processed automatically on first run
```

#### Problem: GEOM dataset issues
**Solutions:**
```bash
# Ensure sufficient disk space (>100GB)
df -h

# Manual download
cd data/geom
wget https://dataverse.harvard.edu/api/access/datafile/4360331
tar -xzvf 4360331

# Process dataset
cd ../..
python build_geom_dataset.py
```

### 3. Memory Issues

#### Problem: `RuntimeError: DataLoader worker (pid(s) X) exited unexpectedly`
**Solution:**
```bash
# Reduce number of workers
python main_qm9.py --num_workers 0  # or 1, 2

# Increase shared memory (Docker)
docker run --shm-size=2g ...
```

#### Problem: System runs out of RAM
**Solutions:**
- Reduce batch size: `--batch_size 16`
- Reduce number of workers: `--num_workers 2`
- Use swap space (not recommended for production)
- Upgrade RAM to 32GB+

## Model Issues

### 1. Poor Training Results

#### Problem: Loss not decreasing
**Solutions:**
- Check learning rate: try `--lr 5e-5` or `--lr 2e-4`
- Verify data preprocessing
- Check model architecture parameters
- Enable gradient clipping
- Increase training epochs

#### Problem: Gradient explosion/vanishing
**Solutions:**
```bash
# Enable gradient clipping
python main_qm9.py --grad_clip True --grad_clip_val 1.0

# Adjust learning rate
python main_qm9.py --lr 1e-5

# Check model normalization
python main_qm9.py --normalize_factors [1,4,10]
```

### 2. Evaluation Issues

#### Problem: Generated molecules are invalid
**Solutions:**
- Check training convergence
- Verify conditioning parameters
- Adjust sampling parameters
- Use pre-trained models for comparison

## Performance Optimization

### 1. Speed Optimization

```bash
# Use multiple GPUs
python main_qm9.py --dp True

# Optimize batch size
python main_qm9.py --batch_size 128  # Adjust based on GPU memory

# Use mixed precision (if supported)
python main_qm9.py --mixed_precision True

# Reduce model size for faster training
python main_qm9.py --nf 128 --n_layers 6
```

### 2. Memory Optimization

```bash
# Gradient checkpointing
python main_qm9.py --gradient_checkpointing True

# Reduce sequence length
python main_qm9.py --max_nodes 20

# Use CPU for some operations
python main_qm9.py --cpu_offload True
```

## Environment-Specific Issues

### 1. Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential libssl-dev libffi-dev

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

### 2. CentOS/RHEL

```bash
# Install dependencies
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel openssl-devel libffi-devel

# For newer versions, use dnf instead of yum
```

### 3. macOS

```bash
# Install dependencies via Homebrew
brew install python@3.10 openbabel

# Note: GPU support not available on macOS
# Use CPU version only
pip install torch torchvision torchaudio
```

### 4. Windows

```bash
# Use Windows Subsystem for Linux (WSL2) recommended
# Or use conda for easier dependency management

conda create -n edm python=3.10
conda activate edm
conda install -c conda-forge rdkit openbabel
```

## Getting Help

### 1. Debug Information

When reporting issues, please include:

```bash
# System information
python verify_installation.py

# Python environment
pip list

# CUDA information (if using GPU)
nvidia-smi
nvcc --version

# Error traceback (full error message)
```

### 2. Useful Commands

```bash
# Check GPU memory usage
nvidia-smi --loop=1

# Monitor training progress
tail -f outputs/your_experiment/log.txt

# Test model loading
python -c "from qm9.models import get_model; print('Model import OK')"

# Verify dataset
python -c "from qm9.dataset import retrieve_dataloaders; print('Dataset OK')"
```

### 3. Common Command Line Arguments

```bash
# Debug mode (small scale)
python main_qm9.py --exp_name debug --n_epochs 1 --batch_size 8 --test_epochs 1

# Memory-efficient training
python main_qm9.py --batch_size 16 --num_workers 2 --nf 128

# CPU-only training
python main_qm9.py --dp False --device cpu

# Resume training
python main_qm9.py --resume outputs/your_experiment
```

## Contact

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/nobkt/e3_diffusion_for_molecules/issues)
2. Search for similar problems in the repository
3. Create a new issue with detailed information including:
   - Operating system and version
   - Python version
   - GPU information (if applicable)
   - Complete error message
   - Steps to reproduce the issue