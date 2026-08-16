#!/usr/bin/env bash
set -euo pipefail

# Generate uv.docker.lock (CPU-only torch) reproducibly.
# Usage: ./scripts/make_docker_lock.sh

TMPDIR=$(mktemp -d)
cp pyproject.toml "$TMPDIR/pyproject.toml"
# Prefer CPU-only PyTorch index for resolution in the temp copy
sed -i 's|https://download.pytorch.org/whl/cu130|https://download.pytorch.org/whl/cpu|g' "$TMPDIR/pyproject.toml"

(cd "$TMPDIR" && uv lock)
cp "$TMPDIR/uv.lock" uv.docker.lock
rm -rf "$TMPDIR"
echo "wrote uv.docker.lock"
