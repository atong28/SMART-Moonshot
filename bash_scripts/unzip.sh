#!/usr/bin/env bash
set -euo pipefail

zip_dir="${1:-MoonshotDatasetv3.zip}"

unzip -q "$DATA_ZIP_DIR/$zip_dir" -d "$DATA_DIR"
