"""Opt-in real PostgreSQL/pgvector integration.

The normal test suite stays independent of Docker. Maintainers can run this
test with ``EMBEDFLOW_RUN_PGVECTOR_DOCKER=1`` to exercise the complete local
container fixture.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest


def test_pgvector_docker_lifecycle():
    if shutil.which("docker") is None:
        pytest.skip("Docker unavailable")
    probe = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f"Docker unavailable: {probe.stderr.strip() or 'daemon is not accessible'}")
    if os.environ.get("EMBEDFLOW_RUN_PGVECTOR_DOCKER") != "1":
        pytest.skip("set EMBEDFLOW_RUN_PGVECTOR_DOCKER=1 to run Docker integration")
    source_example = Path(__file__).parents[1] / "examples" / "pgvector"
    # Copy the example to a temporary directory so generated cache/state files
    # cannot make a later run start WARM.  Keep the checkout root explicit in
    # PYTHONPATH because the helper subprocess runs outside the source tree.
    with tempfile.TemporaryDirectory(prefix="embedflow-pgvector-example-") as raw:
        example = Path(raw) / "pgvector"
        shutil.copytree(source_example, example)
        # Bind an ephemeral host port and use a unique Compose project.  This
        # avoids collisions with a developer's PostgreSQL service or another
        # test invocation while keeping the container's 5432 endpoint fixed.
        compose_text = (example / "compose.yaml").read_text(encoding="utf-8").replace('      - "5432:5432"', '      - "127.0.0.1::5432"')
        (example / "compose.yaml").write_text(compose_text, encoding="utf-8")
        project = f"embedflow-pgvector-{uuid.uuid4().hex[:10]}"
        compose = ["docker", "compose", "-p", project, "-f", str(example / "compose.yaml")]
        env = {
            **os.environ,
            # Set below after Compose reports the ephemeral host port.
            "PYTHONPATH": str(Path(__file__).parents[1]) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        }
        subprocess.run([*compose, "up", "-d"], check=True, capture_output=True, text=True)
        try:
            psycopg = pytest.importorskip("psycopg")
            mapped = subprocess.run([*compose, "port", "postgres", "5432"], check=True, capture_output=True, text=True).stdout.strip()
            host_port = int(mapped.rsplit(":", 1)[1])
            dsn = f"postgresql://embedflow:embedflow@127.0.0.1:{host_port}/embedflow"
            env["EMBEDFLOW_PGVECTOR_DSN"] = dsn
            connection = None
            for _ in range(30):
                try:
                    connection = psycopg.connect(dsn)
                    break
                except psycopg.Error:
                    time.sleep(1)
            if connection is None:
                raise RuntimeError("pgvector container did not become ready")
            connection.close()
            subprocess.run([sys.executable, str(example / "build_index.py")], check=True, cwd=example, env=env)
            smoke = subprocess.run(
                [sys.executable, "-c", "from embedflow.indexes import PgVectorIndex; import numpy as np; i=PgVectorIndex.connect(dsn_env='EMBEDFLOW_PGVECTOR_DSN', dimension=64); print(i.search(np.ones(64, dtype='float32'), 3)); i.close()"],
                check=True, cwd=example, env=env, capture_output=True, text=True,
            )
            assert "SearchHit" in smoke.stdout
            lifecycle = subprocess.run(
                [sys.executable, "-c", """
from embedflow.runtime import open_engine

engine = open_engine('embedflow.yaml', demo=True, start_worker=False)
try:
    cold = engine.search('what causes auroras?', top_k=3, max_sync_misses=0)
    assert cold['migration']['status'] == 'COLD'
    partial = engine.search('what causes auroras?', top_k=3, max_sync_misses=1)
    assert partial['migration']['status'] == 'PARTIAL'
    ids = [hit.document_id for hit in engine.source_index.search(
        engine.source_model.encode_query('what causes auroras?'), 5
    )]
    engine.prewarm(ids, asynchronous=False)
    warm = engine.search('what causes auroras?', top_k=3, max_sync_misses=0)
    assert warm['migration']['status'] == 'WARM'
finally:
    engine.close()

reopened = open_engine('embedflow.yaml', demo=True, start_worker=False)
try:
    assert reopened.search('what causes auroras?', top_k=3, max_sync_misses=0)['migration']['status'] == 'WARM'
    assert reopened.source_index.audit()['ok']
finally:
    reopened.close()
"""],
                check=True, cwd=example, env=env, capture_output=True, text=True,
            )
            assert lifecycle.returncode == 0, lifecycle.stdout + lifecycle.stderr
        finally:
            subprocess.run([*compose, "down", "-v"], check=False, capture_output=True, text=True)
