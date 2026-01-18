#!/usr/bin/env bash
set -euo pipefail
set -x

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
DETACHED_ENVS_DIR="${DETACHED_ENVS_DIR:-/cache/pixi/envs}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/shared/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/shared/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/cache/shared/uv}"

cd "$WORKSPACE_DIR"

if [[ ! -f "pixi.toml" ]]; then
  echo "pixi_install: no pixi.toml in $WORKSPACE_DIR (repo not cloned yet). Done."
  exit 0
fi

mkdir -p "$DETACHED_ENVS_DIR"
pixi config set --system detached-environments "$DETACHED_ENVS_DIR" || true

echo "pixi_install: installing"
pixi --version
pixi install --manifest-path pixi.toml

echo "pixi_install: done"