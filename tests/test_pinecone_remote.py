"""Opt-in, non-destructive Pinecone smoke integration.

This test never creates, upserts, updates, or deletes anything. It targets an
explicit host supplied by the maintainer and is skipped unless the integration
gate is enabled.
"""

from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pytest

from embedflow.indexes import PineconeIndex


def _field(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def test_existing_pinecone_index_smoke():
    if os.environ.get("EMBEDFLOW_PINECONE_INTEGRATION_TEST") != "1":
        pytest.skip("set EMBEDFLOW_PINECONE_INTEGRATION_TEST=1 for remote Pinecone validation")
    if not os.environ.get("PINECONE_API_KEY"):
        pytest.skip("PINECONE_API_KEY unavailable")
    host = os.environ.get("PINECONE_INDEX_HOST")
    if not host:
        pytest.skip("PINECONE_INDEX_HOST unavailable; the remote test never guesses a production index")
    try:
        dimension = int(os.environ.get("PINECONE_INDEX_DIMENSION", "0")) or None
    except ValueError as exc:
        raise AssertionError("PINECONE_INDEX_DIMENSION must be an integer") from exc
    if dimension is None:
        pytest.skip("PINECONE_INDEX_DIMENSION unavailable")
    index = PineconeIndex.connect(
        host=host,
        api_key_env="PINECONE_API_KEY",
        namespace=os.environ.get("PINECONE_NAMESPACE", ""),
        dimension=dimension,
        metric=os.environ.get("PINECONE_INDEX_METRIC", "cosine"),
        text_metadata_field=os.environ.get("PINECONE_TEXT_METADATA_FIELD"),
    )
    try:
        audit = index.audit(source_dimension=dimension)
        assert audit["checks"]["connection"]["ok"]
        assert audit["checks"]["candidate_query"]["ok"]
        vector = np.zeros(dimension, dtype="float32")
        vector[0] = 1.0
        hits = index.search(vector, 1)
        assert len(hits) <= 1
    finally:
        index.close()


def test_temporary_pinecone_index_lifecycle():
    """Opt-in live API test; setup writes are isolated to a disposable index."""

    if os.environ.get("EMBEDFLOW_PINECONE_INTEGRATION_TEST") != "1":
        pytest.skip("set EMBEDFLOW_PINECONE_INTEGRATION_TEST=1 for remote Pinecone validation")
    if os.environ.get("EMBEDFLOW_PINECONE_CREATE_TEST_INDEX") != "1":
        pytest.skip("set EMBEDFLOW_PINECONE_CREATE_TEST_INDEX=1 to create the disposable remote fixture")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        pytest.skip("PINECONE_API_KEY unavailable")
    try:
        import pinecone
    except ImportError:
        pytest.skip('install the optional extra with: pip install "embedflow[pinecone]"')

    client = pinecone.Pinecone(api_key=api_key)
    name = f"embedflow-test-{uuid.uuid4().hex[:12]}"
    namespace = f"embedflow-{uuid.uuid4().hex[:8]}"
    created = False
    index = None
    try:
        spec = pinecone.ServerlessSpec(
            cloud=os.environ.get("PINECONE_TEST_CLOUD", "aws"),
            region=os.environ.get("PINECONE_TEST_REGION", "us-east-1"),
        )
        client.create_index(
            name=name,
            dimension=3,
            metric="cosine",
            spec=spec,
            deletion_protection="disabled",
        )
        created = True
        deadline = time.monotonic() + float(os.environ.get("PINECONE_TEST_TIMEOUT", "180"))
        host = None
        while time.monotonic() < deadline:
            description = client.describe_index(name=name)
            status = _field(description, "status")
            if bool(_field(status, "ready", False)):
                host = _field(description, "host")
                break
            if str(_field(status, "state", "")).lower() == "failed":
                raise AssertionError(f"Pinecone test index {name} failed to become ready")
            time.sleep(2)
        if not host:
            raise AssertionError(f"Pinecone test index {name} was not ready before timeout")

        index = PineconeIndex.connect(
            host=host,
            api_key_env="PINECONE_API_KEY",
            namespace=namespace,
            dimension=3,
            metric="cosine",
            text_metadata_field="text",
        )
        index.index.upsert(
            namespace=namespace,
            vectors=[
                {"id": "north", "values": [1.0, 0.0, 0.0], "metadata": {"text": "north"}},
                {"id": "east", "values": [0.0, 1.0, 0.0], "metadata": {"text": "east"}},
                {"id": "diagonal", "values": [0.7, 0.7, 0.0], "metadata": {"text": "diagonal"}},
            ],
        )
        deadline = time.monotonic() + float(os.environ.get("PINECONE_TEST_TIMEOUT", "180"))
        while time.monotonic() < deadline:
            stats = index.index.describe_index_stats()
            namespaces = _field(stats, "namespaces", {}) or {}
            summary = namespaces.get(namespace) if isinstance(namespaces, dict) else None
            if int(_field(summary, "vector_count", 0) or 0) >= 3:
                break
            time.sleep(2)
        else:
            raise AssertionError(f"Pinecone test namespace {namespace} did not become queryable")

        hits = index.search(np.array([1.0, 0.0, 0.0], dtype="float32"), 3)
        assert [hit.document_id for hit in hits] == ["north", "diagonal", "east"]
        assert index.fetch_documents(["north", "east"]) == {"north": "north", "east": "east"}
        assert index.size() == 3
    finally:
        if index is not None:
            index.close()
        if created:
            try:
                client.delete_index(name=name)
            except Exception as exc:
                raise AssertionError(f"cleanup failed for temporary Pinecone index {name}: {type(exc).__name__}") from exc
