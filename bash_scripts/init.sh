#!/usr/bin/env bash
set -euo pipefail

zip_dir="${1:-MoonshotDatasetv3.zip}"

./pixi_install.sh
./unzip.sh "$zip_dir"