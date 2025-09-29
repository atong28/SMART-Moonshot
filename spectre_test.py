#!/usr/bin/env python
# sweep_test.py
import logging
import os
import sys
import json
import shutil
from datetime import datetime

import pytorch_lightning as pl
import numpy as np
import random
import torch

from src.spectre.core.args import parse_args
from src.spectre.core.settings import SPECTREArgs
from src.spectre.data.fp_loader import EntropyFPLoader
from src.spectre.arch.model import SPECTRE
from src.spectre.test import test
from src.spectre.core.const import DATASET_ROOT, CODE_ROOT
from src.spectre.data.dataset import SPECTREDataModule

# ----------------------------
# Combos (as you specified)
# ----------------------------
COMBO_STRS = [
    #"{hsqc,mw}",
    #"{h_nmr,mw}",
    #"{c_nmr,mw}",
    #"{mass_spec,mw}",
    #"{hsqc,c_nmr,mw}",
    #"{hsqc,h_nmr,mw}",
    #"{hsqc,mass_spec,mw}",
    #"{c_nmr,h_nmr,mw}",
    #"{c_nmr,mass_spec,mw}",
    #"{h_nmr,mass_spec,mw}",
    #"{hsqc,c_nmr,h_nmr,mw}",
    "{hsqc,c_nmr,h_nmr,mass_spec,mw}"
]

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0

def init_logger(path):
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    if not logger.handlers:
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "logs.txt")
        with open(file_path, "w"):
            pass
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_combo_string(s):
    """
    "hsqc" -> ["hsqc"]
    "{hsqc,c_nmr}" -> ["hsqc", "c_nmr"]
    Trims whitespace; preserves order; removes duplicates within a combo.
    """
    s = s.strip()
    if not s:
        return []
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    toks = [t.strip() for t in s.split(",")] if "," in s else [s]
    seen = set()
    out = []
    for t in toks:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def main():
    args: SPECTREArgs = parse_args()
    # We only do testing here; ignore any train flag that slips in
    args.train = False
    args.test = True

    seed_everything(args.seed)

    # Build a results home (one folder for the whole sweep)
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(DATASET_ROOT, "results", args.experiment_name, f"sweep_{today}")
    final_path   = os.path.join(CODE_ROOT,  "results", args.experiment_name, f"sweep_{today}")

    if is_main_process():
        os.makedirs(results_path, exist_ok=True)
        logger = init_logger(results_path)
        logger.info("[Sweep] Parsed args:\n%s", args)
        with open(os.path.join(results_path, "params.json"), "w") as fp:
            json.dump(vars(args), fp, indent=2)
    else:
        logger = None

    # Prepare FP loader once (reused)
    fp_loader = EntropyFPLoader()
    fp_loader.setup(args.out_dim, 6)
    model = SPECTRE(args, fp_loader)

    jsonl_path = os.path.join(results_path, "sweep_results.jsonl")
    csv_path   = os.path.join(results_path, "sweep_metrics.csv")

    # Start fresh
    if is_main_process():
        with open(jsonl_path, "w") as _:
            pass

    rows = []
    # Iterate combos
    for combo_str in COMBO_STRS:
        combo = parse_combo_string(combo_str)
        if not combo:
            continue

        # Per-combo shallow clone of args (no mutation of original object)
        # Since SPECTREArgs is a simple namespace from argparse, direct copy is fine.
        from copy import deepcopy
        local_args = deepcopy(args)
        local_args.input_types = list(combo)
        local_args.requires    = list(combo)

        # Rebuild DataModule/Model so modality wiring is correct
        data_module = SPECTREDataModule(local_args, fp_loader)
        
        if is_main_process():
            logger.info("[Sweep] Testing combo: %s", ",".join(combo))

        # Run test; ask it to emit SWEEP_JSON (you said you'll add sweep=True support)
        res = test(
            local_args, data_module, model, results_path,
            ckpt_path=args.load_from_checkpoint, wandb_run=None, sweep=True
        )

        row = {
            "input_types": ",".join(combo),
            "test/mean_rank_1": res.get("test/mean_rank_1"),
            "test/mean_rank_5": res.get("test/mean_rank_5"),
            "test/mean_rank_10": res.get("test/mean_rank_10"),
        }
        rows.append(row)

        # Also log a JSONL line for auditing / richer info
        if is_main_process():
            payload = {
                "input_types": combo,
                "metrics": {
                    "test/mean_rank_1": row["test/mean_rank_1"],
                    "test/mean_rank_5": row["test/mean_rank_5"],
                    "test/mean_rank_10": row["test/mean_rank_10"],
                },
            }
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps(payload, sort_keys=True) + "\n")

    # Write CSV at the end (rank-0)
    if is_main_process():
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["input_types", "test/mean_rank_1", "test/mean_rank_5", "test/mean_rank_10"]
            )
            w.writeheader()
            w.writerows(rows)
        logger.info("[Sweep] Wrote CSV to %s", csv_path)
        # Optionally mirror to CODE_ROOT like your train path does
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.copytree(results_path, final_path)
        logger.info("[Sweep] Copied sweep folder to %s", final_path)

if __name__ == "__main__":
    main()
