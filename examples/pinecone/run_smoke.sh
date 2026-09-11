#!/usr/bin/env bash
set -euo pipefail

: "${PINECONE_API_KEY:?Set PINECONE_API_KEY first}"
: "${PINECONE_INDEX_HOST:?Set PINECONE_INDEX_HOST to the existing data-plane host}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${script_dir}/../../scripts/pinecone_smoke.py" --host "$PINECONE_INDEX_HOST" --namespace "${PINECONE_NAMESPACE:-}"
