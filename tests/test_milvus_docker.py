"""Opt-in real Milvus standalone integration.

The default suite never requires Docker.  Set ``EMBEDFLOW_RUN_MILVUS_DOCKER=1``
to start an isolated standalone deployment, build two deterministic 10k
fixtures, and run the independent validation harness.  The adapter itself is
never used for fixture writes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest


def _docker_available() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "Docker unavailable"
    probe = subprocess.run(["docker", "info"], capture_output=True, text=True)
    return (probe.returncode == 0, probe.stderr.strip() or "Docker daemon is not accessible")


def test_milvus_docker_10k_lifecycle():
    available, detail = _docker_available()
    if not available:
        pytest.skip(detail)
    if os.environ.get("EMBEDFLOW_RUN_MILVUS_DOCKER") != "1":
        pytest.skip("set EMBEDFLOW_RUN_MILVUS_DOCKER=1 to run the 10k Docker integration")
    repo = Path(__file__).parents[1]
    compose_source = repo / "examples" / "milvus" / "compose.yaml"
    pymilvus_probe = subprocess.run([sys.executable, "-c", "import pymilvus"], capture_output=True, text=True)
    if pymilvus_probe.returncode != 0:
        pytest.skip('install the optional dependency with `python -m pip install "embedflow[milvus]"`')
    project = f"embedflow-milvus-{uuid.uuid4().hex[:10]}"
    hnsw_name = f"ef_milvus_hnsw_{uuid.uuid4().hex[:10]}"
    ivf_name = f"ef_milvus_ivf_{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="embedflow-milvus-docker-") as raw:
        work = Path(raw)
        compose_file = work / "compose.yaml"
        compose_text = compose_source.read_text(encoding="utf-8")
        # Remove fixed container names so parallel test runs and a developer's
        # own example deployment cannot collide.
        compose_text = re.sub(r"^\s+container_name:.*$\n", "", compose_text, flags=re.MULTILINE)
        compose_text = compose_text.replace('"19530:19530"', '"127.0.0.1::19530"')
        compose_text = compose_text.replace('"9091:9091"', '"127.0.0.1::9091"')
        compose_file.write_text(compose_text, encoding="utf-8")
        compose = ["docker", "compose", "-p", project, "-f", str(compose_file)]
        subprocess.run([*compose, "up", "-d"], check=True, capture_output=True, text=True)
        try:
            mapped = subprocess.run([*compose, "port", "standalone", "19530"], check=True, capture_output=True, text=True).stdout.strip()
            port = mapped.rsplit(":", 1)[-1]
            uri = f"http://127.0.0.1:{port}"
            env = {**os.environ, "PYTHONPATH": str(repo) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
            ready = False
            for _ in range(90):
                probe = subprocess.run([sys.executable, "-c", "from pymilvus import MilvusClient; MilvusClient(uri=__import__('os').environ['EMBEDFLOW_MILVUS_URI']).list_collections()"], env={**env, "EMBEDFLOW_MILVUS_URI": uri}, capture_output=True, text=True)
                if probe.returncode == 0:
                    ready = True
                    break
                time.sleep(1)
            if not ready:
                raise RuntimeError("Milvus standalone did not become ready")
            for name, index_type in ((hnsw_name, "HNSW"), (ivf_name, "IVF_FLAT")):
                subprocess.run([sys.executable, str(repo / "scripts" / "milvus_fixture.py"), "--uri", uri,
                                "--collection", name, "--rows", "10000", "--dimension", "64", "--index-type", index_type],
                               check=True, cwd=repo, env=env, timeout=600)
            result = subprocess.run([sys.executable, str(repo / "scripts" / "validate_milvus.py"), "--uri", uri,
                                     "--collection", hnsw_name, "--ivf-collection", ivf_name, "--dimension", "64",
                                     "--cache", str(work / "cache")], check=True, cwd=repo, env=env, capture_output=True, text=True, timeout=900)
            assert '"rows": 10000' in result.stdout
        finally:
            subprocess.run([*compose, "down", "-v"], check=False, capture_output=True, text=True)
