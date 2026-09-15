"""Deterministic, network-free traffic-aware prewarm demonstration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from embedflow.cache import SQLiteVectorCache
from embedflow.models import HashEmbeddingModel
from embedflow.prewarm import PrewarmPlanner, PrewarmRunner
from embedflow.shadow import ShadowTelemetry


class Documents:
    def __init__(self, values: dict[str, str]):
        self.documents = values

    def get(self, ids):
        return {str(document_id): self.documents[str(document_id)] for document_id in ids}

    def size(self):
        return len(self.documents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path(__file__).with_name("runtime"))
    args = parser.parse_args()
    root = args.path.resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    fp = "demo-shadow-fingerprint"
    docs = Documents({f"doc-{i}": f"deterministic document {i}" for i in range(12)})
    telemetry = ShadowTelemetry(root / "shadow.sqlite3", config_fingerprint=fp, max_records=1000)
    target = HashEmbeddingModel("embedflow/prewarm-demo-target", 8)
    cache = SQLiteVectorCache(root / "cache", target.fingerprint, target.dimension)
    try:
        for document_id, count in (("doc-0", 100), ("doc-1", 70), ("doc-2", 20), ("doc-3", 5)):
            for _ in range(count):
                telemetry.record_primary(eligible=True, timestamp=1_000_000)
                telemetry.record_sampled(timestamp=1_000_000)
                telemetry.record_candidate_documents([document_id], timestamp=1_000_000, config_fingerprint=fp)
        cache.put(["doc-0"], np.zeros((1, target.dimension), dtype="float32"),
                  content_fingerprints={"doc-0": cache.content_fingerprint(docs.documents["doc-0"])})
        planner = PrewarmPlanner(telemetry, cache, config_fingerprint=fp,
                                 source_fingerprint="demo-source", target_fingerprint=target.fingerprint,
                                 backend="faiss", index_identity="demo-index", candidate_k=10,
                                 target_dimension=target.dimension, documents=docs)
        plan = planner.plan(start=999_000, end=1_001_000, max_docs=2)
        plan_path = root / "prewarm-plan.json"
        plan_path.write_text(plan.to_json() + "\n", encoding="utf-8")
        runner = PrewarmRunner(target, docs, cache, state_dir=root / "prewarm",
                               source_fingerprint="demo-source", target_fingerprint=target.fingerprint,
                               backend="faiss", index_identity="demo-index", candidate_k=10,
                               config_fingerprint=fp)
        first = runner.run(plan)
        second = runner.run(plan)
        print("PREWARM DEMO")
        print(json.dumps({"selected_ids": plan.selected_ids, "projected_observed_candidate_coverage":
                          plan.selection["estimated_total_candidate_coverage"], "first_run": first,
                          "second_run": second}, indent=2, ensure_ascii=False, default=float))
    finally:
        telemetry.close()
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
