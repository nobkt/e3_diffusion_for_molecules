from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

setup(
    name='e3_diffusion_for_molecules',
    version='1.0.0',
    url='https://github.com/nobkt/e3_diffusion_for_molecules',
    author='E3 Diffusion Team',
    author_email='contact@example.com',
    description='E(3) Equivariant Diffusion Model for Molecule Generation in 3D',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0', 
        'matplotlib>=3.5.0',
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'torchaudio>=0.12.0',
        'ase>=3.22.0',
        'wandb>=0.12.0',
        'tqdm>=4.62.0',
        'imageio>=2.9.0',
        'seaborn>=0.11.0',
        'networkx>=2.6.0',
        'msgpack>=1.0.0',
        'scikit-learn>=1.0.0',
    ],
    extras_require={
        'molecular': ['rdkit>=2022.3.0', 'openbabel-wheel>=3.1.0'],
        'dev': ['pytest>=6.0.0', 'jupyter>=1.0.0'],
        'viz': ['plotly>=5.0.0'],
    },
    python_requires='>=3.8,<3.12',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    keywords='machine-learning deep-learning molecular-generation diffusion-models equivariant pytorch',
)