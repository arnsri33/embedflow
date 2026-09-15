# EmbedFlow v0.8.0 Release Test Report

- Generated: 2026-09-15 13:34:58 -0700
- Python running gate: 3.12.12
- Repository: `embedflow` (local release checkout)
- Git commit: ed8aac8 (working tree has uncommitted changes)

## Gate

**RELEASE GATE: PASS**

## Check matrix

| Check | Status | Detail | Seconds |
|---|---|---|---:|
| public pytest suite | PASS | exit=0 | 22.94 |
| pgvector Docker integration | SKIP | SKIPPED [1] tests/test_pgvector_docker.py:27: Docker unavailable: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock | 0.95 |
| Milvus Docker integration | SKIP | Milvus Docker integration was skipped | 0.93 |
| Weaviate Docker integration | SKIP | SKIPPED [1] tests/test_weaviate_docker.py:32: set EMBEDFLOW_RUN_WEAVIATE_DOCKER=1 to run the 10k Weaviate integration | 0.89 |
| ruff | PASS | exit=0 | 0.06 |
| compileall | PASS | exit=0 | 0.06 |
| registry verification | PASS | exit=0 | 0.23 |
| research regression suite | SKIP | set EMBEDFLOW_RESEARCH_ROOT to a checkout containing tests/ and tests_mvp/ | 0.00 |
| package build | PASS | exit=0 | 2.80 |
| package artifact audit | PASS |  | 0.00 |
| fresh wheel install | PASS | installed wheel and ran help, doctor, registry verify, and demo | 0.00 |
| FAISS example command 1 | PASS | exit=0 | 0.32 |
| FAISS example command 2 | PASS | exit=0 | 0.28 |
| FAISS example command 3 | PASS | exit=0 | 0.28 |
| CLI registry list | PASS | exit=0 | 0.21 |
| CLI registry show | PASS | exit=0 | 0.20 |
| CLI registry match | PASS | exit=0 | 0.20 |
| CLI registry verify | PASS | exit=0 | 0.21 |
| CLI doctor | PASS | exit=0 | 1.84 |
| CLI plan | PASS | exit=0 | 0.30 |
| CLI search | PASS | exit=0 | 0.30 |
| CLI prewarm | PASS | exit=0 | 0.31 |
| CLI audit-index | PASS | exit=0 | 0.29 |
| CLI economics | PASS | exit=0 | 0.17 |
| CLI export-target | PASS | exit=0 | 0.30 |
| CLI migrate | PASS | exit=0 | 0.38 |
| CLI benchmark profiles | PASS | exit=0 | 0.18 |
| Shadow Mode offline demo | PASS | exit=0 | 0.58 |
| CLI Qdrant demo | PASS | exit=0 | 4.80 |
| real local Qdrant model smoke | SKIP | set EMBEDFLOW_MODEL_ROOT to a directory containing minilm_l6/ and qwen3_0_6b/ snapshots | 0.00 |
| FAISS example server | SKIP | sandbox disallows local sockets | 0.00 |
| research example command 1 | PASS | exit=0 | 0.35 |
| research example command 2 | PASS | exit=0 | 0.29 |
| Qdrant example command 1 | PASS | exit=0 | 1.33 |
| Qdrant example command 2 | PASS | exit=0 | 1.39 |
| Qdrant example server | SKIP | sandbox disallows local sockets | 0.00 |
| documentation links | PASS |  | 0.00 |
| security scan | PASS |  | 0.00 |
| manual metadata audit | PASS | none | 0.00 |
| Python 3.10 smoke | SKIP | interpreter unavailable | 0.00 |
| Python 3.11 smoke | SKIP | pytest is not installed in this interpreter | 0.00 |
| Python 3.12 smoke | PASS | exit=0 | 19.59 |

## Test counts

```json
{
  "passed": 199,
  "skipped": 7
}
```

## Python matrix

- Python 3.10: **SKIP**
- Python 3.11: **SKIP**
- Python 3.12: **PASS**

## Registry

Core registry rows: 15
Benchmark profiles: 3
Research summaries: 2
Retained artifacts verified: 0
Semantic values cross-checked: 0
Provenance and packaged checksums are validated by `registry verify`.

## Security

The source scan checks credentials, private keys, absolute local paths, and runtime artifacts. Manual metadata placeholders are a separate release blocker.

## Notes

Expected skips are unavailable interpreters, optional research artifacts/dependencies, Docker or local-socket restrictions, and the opt-in remote Pinecone integration when credentials are absent. No GitHub or external deployment commands are executed by this script.
