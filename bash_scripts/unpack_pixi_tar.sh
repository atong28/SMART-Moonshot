#!/usr/bin/env bash
set -euo pipefail

log() { echo "[unpack_pixi_tar][$(date -Iseconds)] $*"; }
trap 'echo "[unpack_pixi_tar][ERROR] failed at line $LINENO"; exit 1' ERR

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Fast local detached env root (emptyDir)
DETACHED_ROOT="${DETACHED_ROOT:-/cache/pixi/envs}"

# Persistent storage for tarballs (PVC)
PIXI_TAR_DIR="${PIXI_TAR_DIR:-/cache/shared/pixi}"
TARBALL_NAME="${TARBALL_NAME:-pixi-env.tar.zst}"

# Whether to run a fast verify/adopt step after restore
VERIFY_AFTER_RESTORE="${VERIFY_AFTER_RESTORE:-1}"

# Cache locations (PVC-backed) - useful for the verify install
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/shared/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/shared/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/cache/shared/uv}"
export PIXI_HOME="${PIXI_HOME:-/cache/pixi}"

have_zstd() { command -v zstd >/dev/null 2>&1; }

cd "$WORKSPACE_DIR" || true

# Key tarball by lock hash to avoid mismatches
if [[ -f "$WORKSPACE_DIR/pixi.lock" ]]; then
  LOCK_HASH="$(sha256sum "$WORKSPACE_DIR/pixi.lock" | awk '{print $1}')"
  TARBALL_NAME="${TARBALL_NAME/pixi-env/pixi-env-$LOCK_HASH}"
else
  log "WARNING: no pixi.lock found; cannot determine lock-keyed tarball name."
fi

TARBALL_PATH="$PIXI_TAR_DIR/$TARBALL_NAME"
READY_PATH="$TARBALL_PATH.ready"

mkdir -p "$DETACHED_ROOT" "$PIXI_TAR_DIR"

if [[ ! -f "$TARBALL_PATH" || ! -f "$READY_PATH" ]]; then
  log "no ready tarball found at $TARBALL_PATH (or missing $READY_PATH). Skipping restore."
  exit 0
fi

log "restoring detached env folder from $TARBALL_PATH -> $DETACHED_ROOT"

# NOTE: we do NOT rm -rf "$DETACHED_ROOT"/* here anymore because this is a shared detached store.
# If you truly want a clean slate each pod, you can uncomment the next line:
# rm -rf "$DETACHED_ROOT"/*

if [[ "$TARBALL_PATH" == *.tar.zst ]]; then
  if have_zstd; then
    zstd -d -c "$TARBALL_PATH" | tar -C "$DETACHED_ROOT" -xf -
  else
    log "tarball is .tar.zst but zstd is not installed; cannot restore."
    exit 1
  fi
elif [[ "$TARBALL_PATH" == *.tar.gz ]]; then
  tar -C "$DETACHED_ROOT" -xzf "$TARBALL_PATH"
else
  tar -C "$DETACHED_ROOT" -xf "$TARBALL_PATH"
fi

log "restore complete"

if [[ "$VERIFY_AFTER_RESTORE" == "1" && -f "$WORKSPACE_DIR/pixi.toml" ]]; then
  log "verifying/adopting restored env for this workspace (pixi install --locked)"
  pixi --version
  pixi config set --system detached-environments "$DETACHED_ROOT" >/dev/null 2>&1 || true

  if [[ -f "$WORKSPACE_DIR/pixi.lock" ]]; then
    pixi install --locked -v
  else
    pixi install -v
  fi

  log "verify/adopt complete"
fi
