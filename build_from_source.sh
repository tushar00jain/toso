#!/usr/bin/env bash
# One-time (per clean env) build of torchmonarch + torchstore from the in-repo
# source trees into a local uv venv. After this you can edit the source directly:
#   - Python edits (monarch or torchstore) are live — just re-run, no rebuild.
#   - Rust edits (monarch) need a recompile; re-run rebuild_monarch.sh, which is
#     incremental because CARGO_TARGET_DIR persists (cargo only rebuilds what
#     changed).
#
# Must be run by a human, not the agent: it needs Rust (installed here via rustup)
# and cargo pulls crates over the network. Run it yourself, then tell the agent.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

# --- build knobs -------------------------------------------------

# Persistent cargo target dir => incremental Rust rebuilds across runs.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/monarch-cargo-target}"

# CPU-only build: skip the libtorch C++ tensor engine and all GPU/NCCL/RDMA.
export USE_TENSOR_ENGINE="${USE_TENSOR_ENGINE:-0}"
export MONARCH_GPU_PLATFORM="${MONARCH_GPU_PLATFORM:-none}"

# --- 1. Rust nightly (monarch's RUSTFLAGS require nightly) -----------------
if ! command -v cargo >/dev/null 2>&1; then
  echo ">>> Installing Rust (rustup, nightly default)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain nightly
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
rustup toolchain install nightly >/dev/null 2>&1 || true
rustup default nightly

# --- 2. venv + build/runtime deps (needed before --no-build-isolation) -----
echo ">>> Creating venv and installing build + base deps..."
uv venv --python 3.12 --clear
uv pip install setuptools setuptools-rust wheel numpy torch pygtrie portpicker

# --- 3. editable source builds (reuse env torch; incremental via cargo) ----
echo ">>> Building torchmonarch from source (first build compiles ~800 crates)..."
uv pip install --no-build-isolation -e ../monarch

echo ">>> Building torchstore from source..."
uv pip install --no-build-isolation -e ../torchstore

echo
echo "Done. Run the live example with:"
echo "  uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 live_example.py --port 8099"
