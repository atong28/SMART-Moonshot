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

## Step 1: Refactor and reproduce SPECTRE results.

If you have the old version of the dataset, you can reformat the dataset with `scripts/restructure_dataset.py`. It will work even without the simulated MS data or the ID data, but you can find the scripts in the old SPECTRE repo under their respective branches.

Settings are located in `src/settings.py` and can be overridden using cmdline arguments in `main.py`. `main.py` is the planned entrypoint for all the code and designed to be able to be run with only default args, updating the args as needed.

Support for Entropy FP is tested and working, and HYUN FP should be allowed but not fully tested yet. We resort to only Ranked Transformer models and no longer use old code in the `xwd0418/Spectre` repo.

# Cross Attention Model
Extremely basic change from SPECTRE: replace the self attention blocks with cross attention on a CLS token explicitly.

# Mixed Attention Model
Add self attention per-modality and then cross attend to CLS tokens. Finally, encode sequence of CLS tokens with self attention into the first CLS token which adds molecular weight as a modality.

# Mixed Attention 2 Model
Identical to mixed attention model, except instead of appending onto the first CLS token (often hsqc) we use a global token. This is useful for consistency once we introduce all optional inputs (where hsqc might be omitted)

# Mixed Attention 3 Model
We actually implement it now so that we don't need any required inputs. Dataset is updated for this adjustment, and increase number of layers. Remove isotopic distribution as well. Also, now molecular weight is added before transformer encoder layer.

