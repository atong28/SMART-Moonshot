# SMART-Moonshot Project

## Installation

Install pixi according to the following instructions:
```
https://pixi.sh/dev/installation/
```
If you are not running on linux-64, you can try adding your distro into `pixi.toml` and install anyways, but no guarantees for support. Running the following command should automatically boot you into the shell with the loaded environment:
```
pixi shell
```
To just install the environment, use
```
pixi install
```

## Dataset setup

```
PYTHONPATH=src python -m spectre.data.fp_loader fragments --index ~/MoonshotDatasetv3/index.pkl --out-dir ~/MoonshotDatasetv3
```