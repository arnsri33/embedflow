# EmbedFlow v0.7.0 Release Test Report

- Generated: 2026-09-14 12:44:22 -0700
- Python running gate: 3.12.12
- Repository: `embedflow` (local release checkout)
- Git commit: 9d9e8eb (working tree has uncommitted changes)

## Gate

**RELEASE GATE: PASS**

## Check matrix

| Check | Status | Detail | Seconds |
|---|---|---|---:|
| public pytest suite | PASS | exit=0 | 10.95 |
| pgvector Docker integration | SKIP | SKIPPED [1] tests/test_pgvector_docker.py:27: Docker unavailable: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock | 0.93 |
| Milvus Docker integration | SKIP | Milvus Docker integration was skipped | 0.92 |
| Weaviate Docker integration | SKIP | SKIPPED [1] tests/test_weaviate_docker.py:32: set EMBEDFLOW_RUN_WEAVIATE_DOCKER=1 to run the 10k Weaviate integration | 0.86 |
| ruff | PASS | exit=0 | 0.05 |
| compileall | PASS | exit=0 | 0.06 |
| registry verification | PASS | exit=0 | 0.07 |
| research regression suite | SKIP | set EMBEDFLOW_RESEARCH_ROOT to a checkout containing tests/ and tests_mvp/ | 0.00 |
| package build | PASS | exit=0 | 2.73 |
| package artifact audit | PASS |  | 0.00 |
| fresh wheel install | PASS | installed wheel and ran help, doctor, registry verify, and demo | 0.00 |
| FAISS example command 1 | PASS | exit=0 | 0.33 |
| FAISS example command 2 | PASS | exit=0 | 0.27 |
| FAISS example command 3 | PASS | exit=0 | 0.31 |
| CLI registry list | PASS | exit=0 | 0.19 |
| CLI registry show | PASS | exit=0 | 0.18 |
| CLI registry match | PASS | exit=0 | 0.22 |
| CLI registry verify | PASS | exit=0 | 0.20 |
| CLI doctor | PASS | exit=0 | 1.87 |
| CLI plan | PASS | exit=0 | 0.30 |
| CLI search | PASS | exit=0 | 0.35 |
| CLI prewarm | PASS | exit=0 | 0.31 |
| CLI audit-index | PASS | exit=0 | 0.24 |
| CLI economics | PASS | exit=0 | 0.18 |
| CLI export-target | PASS | exit=0 | 0.30 |
| CLI migrate | PASS | exit=0 | 0.36 |
| CLI benchmark profiles | PASS | exit=0 | 0.18 |
| Shadow Mode offline demo | PASS | exit=0 | 0.58 |
| CLI Qdrant demo | PASS | exit=0 | 4.71 |
| real local Qdrant model smoke | SKIP | set EMBEDFLOW_MODEL_ROOT to a directory containing minilm_l6/ and qwen3_0_6b/ snapshots | 0.00 |
| FAISS example server | SKIP | sandbox disallows local sockets | 0.00 |
| research example command 1 | PASS | exit=0 | 0.36 |
| research example command 2 | PASS | exit=0 | 0.30 |
| Qdrant example command 1 | PASS | exit=0 | 1.37 |
| Qdrant example command 2 | PASS | exit=0 | 1.35 |
| Qdrant example server | SKIP | sandbox disallows local sockets | 0.00 |
| documentation links | PASS |  | 0.00 |
| security scan | PASS |  | 0.00 |
| manual metadata audit | PASS | none | 0.00 |
| Python 3.10 smoke | SKIP | interpreter unavailable | 0.00 |
| Python 3.11 smoke | SKIP | pytest is not installed in this interpreter | 0.00 |
| Python 3.12 smoke | PASS | exit=0 | 10.99 |

## Test counts

```json
{
  "passed": 186,
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
