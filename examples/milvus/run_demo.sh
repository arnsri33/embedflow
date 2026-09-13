#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/milvus_fixture.py" --uri "${EMBEDFLOW_MILVUS_URI:-http://127.0.0.1:19530}" --collection "${EMBEDFLOW_MILVUS_COLLECTION:-embedflow_demo}"
cp "$DEMO_DIR/embedflow.yaml.example" "$DEMO_DIR/embedflow.yaml"
EMBEDFLOW_MILVUS_URI="${EMBEDFLOW_MILVUS_URI:-http://127.0.0.1:19530}" \
EMBEDFLOW_MILVUS_COLLECTION="${EMBEDFLOW_MILVUS_COLLECTION:-embedflow_demo}" \
  "$PYTHON_BIN" -m embedflow doctor --config "$DEMO_DIR/embedflow.yaml"
EMBEDFLOW_MILVUS_URI="${EMBEDFLOW_MILVUS_URI:-http://127.0.0.1:19530}" \
EMBEDFLOW_MILVUS_COLLECTION="${EMBEDFLOW_MILVUS_COLLECTION:-embedflow_demo}" \
  "$PYTHON_BIN" -m embedflow audit-index --config "$DEMO_DIR/embedflow.yaml"
