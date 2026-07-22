#!/usr/bin/env bash
# Recompile Monarch's Rust extension after editing its Rust source. Incremental:
# cargo reuses $CARGO_TARGET_DIR and only rebuilds the crates you touched.
#
# You do NOT need this for Python-only edits (monarch or torchstore) — those are
# picked up live from the editable install.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/monarch-cargo-target}"
export USE_TENSOR_ENGINE="${USE_TENSOR_ENGINE:-0}"
export MONARCH_GPU_PLATFORM="${MONARCH_GPU_PLATFORM:-none}"

# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

uv pip install --no-build-isolation -e ../monarch
echo "Rebuilt torchmonarch."
