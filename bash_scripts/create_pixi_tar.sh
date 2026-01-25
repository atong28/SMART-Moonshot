#!/usr/bin/env bash
set -euo pipefail

log() { echo "[create_pixi_tar][$(date -Iseconds)] $*"; }
trap 'echo "[create_pixi_tar][ERROR] failed at line $LINENO"; exit 1' ERR

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Fast local detached env root (emptyDir)
DETACHED_ROOT="${DETACHED_ROOT:-/cache/pixi/envs}"

# Persistent storage for tarballs (PVC)
PIXI_TAR_DIR="${PIXI_TAR_DIR:-/cache/shared/pixi}"
TARBALL_NAME="${TARBALL_NAME:-pixi-env.tar.zst}"

# Cache locations (PVC-backed)
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/shared/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/shared/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/cache/shared/uv}"
export PIXI_HOME="${PIXI_HOME:-/cache/pixi}"  # OK if emptyDir; not critical

have_zstd() { command -v zstd >/dev/null 2>&1; }

cd "$WORKSPACE_DIR"

if [[ ! -f "pixi.toml" ]]; then
  log "no pixi.toml in $WORKSPACE_DIR (repo not cloned yet). Done."
  exit 0
fi

# Key tarball by lock hash to avoid mismatches
if [[ -f "pixi.lock" ]]; then
  LOCK_HASH="$(sha256sum "pixi.lock" | awk '{print $1}')"
  TARBALL_NAME="${TARBALL_NAME/pixi-env/pixi-env-$LOCK_HASH}"
else
  log "WARNING: no pixi.lock found; tarball will not be lock-keyed."
fi

TARBALL_PATH="$PIXI_TAR_DIR/$TARBALL_NAME"
READY_PATH="$TARBALL_PATH.ready"
TMP_PATH="$TARBALL_PATH.tmp"

mkdir -p "$DETACHED_ROOT" "$PIXI_TAR_DIR"

log "installing env into detached env root: $DETACHED_ROOT"
pixi --version
pixi config set --system detached-environments "$DETACHED_ROOT" >/dev/null 2>&1 || true

if [[ -f "pixi.lock" ]]; then
  log "pixi.lock found; running: pixi install --locked"
  pixi install --locked -v
else
  log "no pixi.lock; running: pixi install (solver may run)"
  pixi install -v
fi

# Identify the *project folder* inside DETACHED_ROOT by using the symlink Pixi creates in the workspace
# /workspace/.pixi/envs -> /cache/pixi/envs/<PROJECT_ID>/envs
ENV_DIR="$(readlink -f "$WORKSPACE_DIR/.pixi/envs")"
PROJECT_DIR="$(dirname "$ENV_DIR")"                 # /cache/pixi/envs/<PROJECT_ID>
PROJECT_BASENAME="$(basename "$PROJECT_DIR")"       # <PROJECT_ID>

# Sanity checks to avoid packing the wrong thing
case "$PROJECT_DIR" in
  "$DETACHED_ROOT"/*) ;;
  *)
    log "ERROR: resolved PROJECT_DIR is not under DETACHED_ROOT"
    log "  DETACHED_ROOT=$DETACHED_ROOT"
    log "  ENV_DIR=$ENV_DIR"
    log "  PROJECT_DIR=$PROJECT_DIR"
    exit 1
    ;;
esac

if [[ ! -d "$PROJECT_DIR" ]]; then
  log "ERROR: project dir does not exist: $PROJECT_DIR"
  exit 1
fi

log "packing project detached env folder: $PROJECT_DIR"
log "  -> $TARBALL_PATH"
rm -f "$READY_PATH" "$TMP_PATH"

# Pack only the project folder (so tarball stays reusable & doesn't include unrelated envs)
if [[ "$TARBALL_PATH" == *.tar.zst ]]; then
  if have_zstd; then
    tar -C "$DETACHED_ROOT" -cf - "$PROJECT_BASENAME" | zstd -3 -T0 -o "$TMP_PATH"
  else
    log "TARBALL_NAME ends with .tar.zst but zstd is not installed."
    exit 1
  fi
elif [[ "$TARBALL_PATH" == *.tar.gz ]]; then
  tar -C "$DETACHED_ROOT" -czf "$TMP_PATH" "$PROJECT_BASENAME"
else
  tar -C "$DETACHED_ROOT" -cf "$TMP_PATH" "$PROJECT_BASENAME"
fi

mv -f "$TMP_PATH" "$TARBALL_PATH"
touch "$READY_PATH"

log "tarball ready: $TARBALL_PATH"
