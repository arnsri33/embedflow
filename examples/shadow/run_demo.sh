#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/embedflow-shadow-demo.XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT

cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" examples/shadow/run_demo.py --path "$RUN_DIR"
