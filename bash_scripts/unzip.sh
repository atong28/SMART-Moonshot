#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ZIP_DIR:?DATA_ZIP_DIR is not set}"
: "${DATA_DIR:?DATA_DIR is not set}"

ZIP_PATH="$DATA_ZIP_DIR/$1"

echo "[unzip] zip=$ZIP_PATH"
echo "[unzip] dest=$DATA_DIR"

ls -lh "$ZIP_PATH"

unzip -q "$ZIP_PATH" -d "$DATA_DIR"

echo "[unzip] done"
