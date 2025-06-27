# SMART-Moonshot Project

## Step 1: Refactor and reproduce SPECTRE results.

If you have the old version of the dataset, you can reformat the dataset with `scripts/restructure_dataset.py`. It will work even without the simulated MS data or the ID data, but you can find the scripts in the old SPECTRE repo under their respective branches.

Settings are located in `src/settings.py` and can be overridden using cmdline arguments in `main.py`. `main.py` is the planned entrypoint for all the code and designed to be able to be run with only default args, updating the args as needed.

Support for Entropy FP is tested and working, and HYUN FP should be allowed but not fully tested yet. We resort to only Ranked Transformer models and no longer use old code in the `xwd0418/Spectre` repo.