#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./launch.sh <marina|spectre> [--nproc_per_node N] [--] [args...]

Behavior:
  - Mirrors: torchrun --nproc_per_node=4 --module src.main <arch> [args...]
  - Defaults match pixi task "train.marina":
      CUDA_LAUNCH_BLOCKING=1
      --nproc_per_node=4
  - Creates the run directory at (matches get_data_paths.py final_path):
      $PVC_ROOT/results/<experiment_name>/<RUN_ID>
    and redirects stdout/stderr to:
      stdout_stderr.log

Notes:
  - Override defaults via env:
      NPROC_PER_NODE=2 CUDA_LAUNCH_BLOCKING=0 ./launch.sh marina
  - To control the results directory name deterministically:
      SMART_RUN_ID=2026-02-24_12-00-00 ./launch.sh marina
  - To override experiment name:
      ./launch.sh marina -- --experiment_name my-exp
EOF
}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

arch="${1:-}"
if [[ -z "$arch" || "$arch" == "-h" || "$arch" == "--help" ]]; then
  usage
  exit 0
fi
shift

case "$arch" in
  marina|spectre) ;;
  *)
    echo "[launch.sh] ERROR: arch must be 'marina' or 'spectre' (got: $arch)" >&2
    exit 2
    ;;
esac

nproc_per_node="${NPROC_PER_NODE:-4}"
cuda_launch_blocking="${CUDA_LAUNCH_BLOCKING:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc_per_node=*)
      nproc_per_node="${1#*=}"
      shift
      ;;
    --nproc_per_node)
      nproc_per_node="${2:?missing value for --nproc_per_node}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

default_experiment_name=""
case "$arch" in
  marina) default_experiment_name="marina-development" ;;
  spectre) default_experiment_name="spectre-development" ;;
esac

experiment_name="$default_experiment_name"
forward_args=("$@")
for ((i=0; i<${#forward_args[@]}; i++)); do
  a="${forward_args[i]}"
  if [[ "$a" == --experiment_name=* ]]; then
    experiment_name="${a#*=}"
  elif [[ "$a" == --experiment_name ]]; then
    if (( i + 1 < ${#forward_args[@]} )); then
      experiment_name="${forward_args[i+1]}"
    fi
  fi
done

run_id="${SMART_RUN_ID:-$(date +"%Y-%m-%d_%H-%M-%S")}"
export SMART_RUN_ID="$run_id"
export CUDA_LAUNCH_BLOCKING="$cuda_launch_blocking"
export PYTHONUNBUFFERED=1

if [[ -d "/root/gurusmart" ]]; then
  pvc_root="/root/gurusmart/Moonshot"
  export LD_LIBRARY_PATH="/code/.pixi/envs/default/lib:${LD_LIBRARY_PATH-}"
else
  pvc_root="/data/nas-gpu/wang/atong/SMART-Moonshot"
  export LD_LIBRARY_PATH="/data/nas-gpu/wang/atong/SMART-Moonshot/.pixi/envs/default/lib:${LD_LIBRARY_PATH-}"
fi

final_dir="${pvc_root%/}/results/${experiment_name}/${run_id}"
mkdir -p "$final_dir"
log_file="${final_dir}/stdout_stderr.log"

progress_regex='^((Epoch|Validation DataLoader|Testing DataLoader) [0-9]+:|Benchmarking|Testing|Validation|Sanity Checking|Epoch)'
# Keep full output on console, but don't persist Lightning progress-bar lines.
exec > >(tee >(tr -d '\r' | grep -Ev "$progress_regex" >> "$log_file")) 2>&1

echo "[launch.sh] arch=$arch"
echo "[launch.sh] nproc_per_node=$nproc_per_node"
echo "[launch.sh] CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "[launch.sh] SMART_RUN_ID=$SMART_RUN_ID"
echo "[launch.sh] PVC_ROOT=$pvc_root"
echo "[launch.sh] final_dir=$final_dir"
echo "[launch.sh] log_file=$log_file"

pixi run torchrun --nproc_per_node="$nproc_per_node" --module src.main "$arch" "${forward_args[@]}"

