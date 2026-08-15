#!/usr/bin/env bash
set -euo pipefail

export DENO_INSTALL="${DENO_INSTALL:-/opt/render/project/.deno}"
curl -fsSL https://deno.land/install.sh | sh
python -m pip install -r requirements.txt
