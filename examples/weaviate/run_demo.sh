#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose -f compose.yaml up -d
python ../../scripts/weaviate_fixture.py --uri http://127.0.0.1:8080 --collection Documents --rows "${WEAVIATE_ROWS:-100}"
cp -n embedflow.yaml.example embedflow.yaml 2>/dev/null || true
python ../../scripts/weaviate_smoke.py --uri http://127.0.0.1:8080 --collection Documents
