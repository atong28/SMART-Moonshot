# SMART-Moonshot Project

## Installation

```
conda env create -n moonshot python=3.12
conda activate moonshot
pip install torch==2.5.1+cu124 \
            torchvision==0.20.1+cu124 \
            torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu124
conda env update -n moonshot -f environment.yml --prune
```

## Dataset setup

```
PYTHONPATH=src python -m spectre.data.fp_loader fragments --index ~/MoonshotDatasetv3/index.pkl --out-dir ~/MoonshotDatasetv3
```