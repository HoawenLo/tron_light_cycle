#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Check destination compatibility.
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "This bundle requires Ubuntu/Linux x86-64."
    exit 1
fi

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Python 3.12 is required."
    exit 1
fi

# Unpack bundled libraries on first launch.
if [[ ! -f vendor/.complete ]]; then
    mkdir -p vendor
    cat vendor.tar.gz.part-* | tar -xz -C vendor
    touch vendor/.complete
fi

export PYTHONPATH="$PWD/vendor:$PWD/app/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

exec python3.12 -S -m tron_ai.play \
  --checkpoint models/dqn_intermediate.npz \
  --checkpoint models/ppo_advanced.npz \
  --opponent ppo \
  --speed 6 \
  "$@"
