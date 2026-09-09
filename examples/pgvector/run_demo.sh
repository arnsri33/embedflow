#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EMBEDFLOW_PGVECTOR_DSN="${EMBEDFLOW_PGVECTOR_DSN:-postgresql://embedflow:embedflow@localhost:5432/embedflow}"

python "${example_dir}/build_index.py"
embedflow analyze --config "${example_dir}/embedflow.yaml" --demo --output-dir "${example_dir}/runtime/analysis"
echo
echo "Starting EmbedFlow at http://127.0.0.1:8000/ (Ctrl-C to stop)"
exec embedflow serve --config "${example_dir}/embedflow.yaml" --demo --host 127.0.0.1 --port 8000
