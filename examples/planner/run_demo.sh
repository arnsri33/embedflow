#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/embedflow-planner-demo.XXXXXX")"
trap 'rm -rf "$RUN_DIR"' EXIT

cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m embedflow demo --path "$RUN_DIR" --backend faiss --no-serve
"$PYTHON_BIN" -m embedflow plan \
  --config "$RUN_DIR/embedflow.yaml" \
  --queries "$RUN_DIR/probe_queries.jsonl" \
  --demo
