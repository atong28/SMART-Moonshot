"""
Python launcher for SMART-Moonshot training/eval.

Runs torchrun directly via the current Python interpreter (no bash, no pixi run):

  python launch.py <marina|spectre|diffms> [--nproc_per_node N] [--] [args...]

Equivalent to:
  python -m torch.distributed.run --nproc_per_node=4 --module src.main <arch> [args...]

Defaults (matching pixi task "train.marina"):
  CUDA_LAUNCH_BLOCKING=1, --nproc_per_node=4

Creates the run directory at (matches get_data_paths.py final_path):
  $PVC_ROOT/results/<experiment_name>/<RUN_ID>
When ENABLE_STDOUT_LOG=1, tees combined stdout/stderr to stdout_stderr.log.

Environment overrides:
  NPROC_PER_NODE=2 CUDA_LAUNCH_BLOCKING=0 python launch.py marina
  SMART_RUN_ID=2026-02-24_12-00-00 python launch.py marina
  ENABLE_STDOUT_LOG=1 python launch.py marina
  python launch.py marina -- --experiment_name my-exp
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


VALID_ARCHES = {"marina", "spectre", "diffms"}


USAGE = """Usage:
  python launch.py <marina|spectre|diffms> [--nproc_per_node N] [--] [args...]

Behavior:
  - Runs: python -m torch.distributed.run --nproc_per_node=4 --module src.main <arch> [args...]
  - Defaults: CUDA_LAUNCH_BLOCKING=1, --nproc_per_node=4
  - Creates run dir: $PVC_ROOT/results/<experiment_name>/<RUN_ID>
  - With ENABLE_STDOUT_LOG=1: stdout/stderr also written to stdout_stderr.log

Notes:
  - NPROC_PER_NODE=2 CUDA_LAUNCH_BLOCKING=0 python launch.py marina
  - SMART_RUN_ID=2026-02-24_12-00-00 python launch.py marina
  - python launch.py marina -- --experiment_name my-exp
"""


def parse_cli(argv: List[str]) -> Tuple[str, str, str, List[str]]:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, file=sys.stdout)
        sys.exit(0)

    arch = argv[0]
    if arch not in VALID_ARCHES:
        print(
            f"[launch.py] ERROR: arch must be 'marina' or 'spectre' or 'diffms' (got: {arch})",
            file=sys.stderr,
        )
        sys.exit(2)

    nproc_per_node = os.environ.get("NPROC_PER_NODE", "4")
    cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "1")

    rest = argv[1:]
    forward_args: List[str] = []
    i = 0
    while i < len(rest):
        a = rest[i]
        if a.startswith("--nproc_per_node="):
            nproc_per_node = a.split("=", 1)[1]
            i += 1
        elif a == "--nproc_per_node":
            if i + 1 >= len(rest):
                print("[launch.py] ERROR: missing value for --nproc_per_node", file=sys.stderr)
                sys.exit(2)
            nproc_per_node = rest[i + 1]
            i += 2
        elif a == "--":
            forward_args.extend(rest[i + 1 :])
            break
        else:
            forward_args.extend(rest[i:])
            break

    return arch, nproc_per_node, cuda_launch_blocking, forward_args


def resolve_experiment_name(arch: str, forward_args: List[str]) -> str:
    if arch == "marina":
        experiment_name = "marina-development"
    elif arch == "spectre":
        experiment_name = "spectre-development"
    else:
        experiment_name = "diffms-development"

    i = 0
    while i < len(forward_args):
        a = forward_args[i]
        if a.startswith("--experiment_name="):
            experiment_name = a.split("=", 1)[1]
        elif a == "--experiment_name" and i + 1 < len(forward_args):
            experiment_name = forward_args[i + 1]
            i += 1
        i += 1

    return experiment_name


def compute_paths(experiment_name: str) -> tuple[str, str, str]:
    if os.path.isdir("/root/gurusmart"):
        pvc_root = "/root/gurusmart/Moonshot"
        ld_prefix = "/code/.pixi/envs/default/lib"
    else:
        pvc_root = "/data/nas-gpu/wang/atong/SMART-Moonshot"
        ld_prefix = "/data/nas-gpu/wang/atong/SMART-Moonshot/.pixi/envs/default/lib"

    run_id = os.environ.get("SMART_RUN_ID") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.environ["SMART_RUN_ID"] = run_id

    final_dir = f"{pvc_root.rstrip('/')}/results/{experiment_name}/{run_id}"
    log_file = os.path.join(final_dir, "stdout_stderr.log")

    prev_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if prev_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{ld_prefix}:{prev_ld}"
    else:
        os.environ["LD_LIBRARY_PATH"] = ld_prefix

    os.environ["PYTHONUNBUFFERED"] = "1"

    return pvc_root, final_dir, log_file


def log_and_stream_process(cmd: List[str], log_file: str, env: dict) -> int:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a", buffering=1, encoding="utf-8") as f:
        def _log_line(line: str) -> None:
            clean = line.replace("\r", "")
            print(clean, end="", flush=True)
            f.write(clean)
            f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            _log_line(line)

        return proc.wait()


def main(argv: List[str]) -> int:
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    arch, nproc_per_node, cuda_launch_blocking, forward_args = parse_cli(argv)

    os.environ["CUDA_LAUNCH_BLOCKING"] = str(cuda_launch_blocking)

    experiment_name = resolve_experiment_name(arch, forward_args)
    pvc_root, final_dir, log_file = compute_paths(experiment_name)

    enable_stdout_log = os.environ.get("ENABLE_STDOUT_LOG", "1") == "1"

    # Run torchrun via current Python interpreter (no bash, no pixi).
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc_per_node),
        "--module",
        "src.main",
        arch,
        *forward_args,
    ]

    child_env = os.environ.copy()

    def _print_launch_config(prefix: str) -> None:
        print(f"{prefix} arch={arch}")
        print(f"{prefix} nproc_per_node={nproc_per_node}")
        print(f"{prefix} CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
        print(f"{prefix} SMART_RUN_ID={os.environ.get('SMART_RUN_ID')}")
        print(f"{prefix} PVC_ROOT={pvc_root}")
        print(f"{prefix} final_dir={final_dir}")
        print(f"{prefix} log_file={log_file}")
        sys.stdout.flush()

    os.makedirs(final_dir, exist_ok=True)

    if enable_stdout_log:
        with open(log_file, "a", buffering=1, encoding="utf-8") as f:
            for line in [
                "[launch.py] stdout/stderr logging enabled (ENABLE_STDOUT_LOG=1)",
                f"[launch.py] arch={arch}",
                f"[launch.py] nproc_per_node={nproc_per_node}",
                f"[launch.py] CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}",
                f"[launch.py] SMART_RUN_ID={os.environ.get('SMART_RUN_ID')}",
                f"[launch.py] PVC_ROOT={pvc_root}",
                f"[launch.py] final_dir={final_dir}",
                f"[launch.py] log_file={log_file}",
            ]:
                print(line, flush=True)
                f.write(line + "\n")
        return log_and_stream_process(cmd, log_file, child_env)
    else:
        print("[launch.py] stdout/stderr logging disabled (ENABLE_STDOUT_LOG=0)")
        _print_launch_config("[launch.py]")
        proc = subprocess.Popen(cmd, env=child_env)
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
