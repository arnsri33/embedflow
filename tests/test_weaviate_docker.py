"""Opt-in real Weaviate v4 integration.

The ordinary suite remains credential- and Docker-free.  Set
``EMBEDFLOW_RUN_WEAVIATE_DOCKER=1`` to create an isolated 10k externally
vectorized fixture, exercise the real client/adapter, and remove the
temporary collections and container afterward.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_weaviate_docker_10k_lifecycle(tmp_path: Path):
    if shutil.which("docker") is None:
        pytest.skip("Docker unavailable")
    if os.environ.get("EMBEDFLOW_RUN_WEAVIATE_DOCKER") != "1":
        pytest.skip("set EMBEDFLOW_RUN_WEAVIATE_DOCKER=1 to run the 10k Weaviate integration")
    probe = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(probe.stderr.strip() or "Docker daemon is not accessible")
    try:
        import weaviate  # noqa: F401
    except ImportError:
        pytest.skip('install the optional dependency with `python -m pip install "embedflow[weaviate]"`')
    repo = Path(__file__).parents[1]
    project = f"embedflow-weaviate-{uuid.uuid4().hex[:10]}"
    http_port, grpc_port = _free_port(), _free_port()
    collection = f"EmbedFlow10k{uuid.uuid4().hex[:10]}"
    env = {**os.environ, "WEAVIATE_HTTP_PORT": str(http_port), "WEAVIATE_GRPC_PORT": str(grpc_port)}
    compose = ["docker", "compose", "-p", project, "-f", str(repo / "examples" / "weaviate" / "compose.yaml")]
    subprocess.run([*compose, "up", "-d"], check=True, capture_output=True, text=True, env=env)
    try:
        uri = f"http://127.0.0.1:{http_port}"
        ready = False
        for _ in range(120):
            check = subprocess.run([sys.executable, "-c", "import weaviate; c=weaviate.connect_to_custom(http_host='127.0.0.1',http_port=int(__import__('os').environ['W_HTTP']),http_secure=False,grpc_host='127.0.0.1',grpc_port=int(__import__('os').environ['W_GRPC']),grpc_secure=False); print(c.is_ready()); c.close()"],
                                    env={**env, "W_HTTP": str(http_port), "W_GRPC": str(grpc_port)},
                                    capture_output=True, text=True)
            if check.returncode == 0 and "True" in check.stdout:
                ready = True
                break
            import time
            time.sleep(1)
        if not ready:
            raise RuntimeError("Weaviate did not become ready")
        subprocess.run([sys.executable, str(repo / "scripts" / "weaviate_fixture.py"), "--uri", uri,
                        "--http-port", str(http_port), "--grpc-port", str(grpc_port), "--collection", collection,
                        "--rows", "10000", "--dimension", "64"], check=True, cwd=repo, env={**env, "PYTHONPATH": str(repo)}, timeout=900)
        subprocess.run([sys.executable, str(repo / "scripts" / "validate_weaviate.py"), "--uri", uri,
                        "--http-port", str(http_port), "--grpc-port", str(grpc_port), "--collection", collection,
                        "--rows", "10000", "--dimension", "64", "--cache", str(tmp_path / "cache")],
                       check=True, cwd=repo, env={**env, "PYTHONPATH": str(repo)}, timeout=900)
    finally:
        subprocess.run([*compose, "down", "-v"], check=False, capture_output=True, text=True, env=env)
