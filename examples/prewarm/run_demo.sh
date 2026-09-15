#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# Make the checkout runnable without a prior editable install, while leaving a
# clean site-packages installation authoritative when this directory is copied
# outside the repository for the wheel smoke test.
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
if [ -d "$PROJECT_ROOT/embedflow" ]; then
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONPATH
fi
exec python3 "$SCRIPT_DIR/demo.py" "$@"
