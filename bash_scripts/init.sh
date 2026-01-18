#!/usr/bin/env bash
set -euo pipefail

zip_dir="${1:-MoonshotDatasetv3.zip}"

$WORKSPACE_DIR/bash_scripts/unpack_pixi_tar.sh
$WORKSPACE_DIR/bash_scripts/unzip.sh "$zip_dir"