# EmbedFlow v0.1.0 Release Test Report

- Generated: 2026-09-07 16:20:25 -0700
- Python running gate: 3.12.12
- Repository: `embedflow` (local release checkout)
- Git commit: 8e79ba4 (working tree has uncommitted changes)

## Gate

**RELEASE GATE: PASS**

## Check matrix

| Check | Status | Detail | Seconds |
|---|---|---|---:|
| public pytest suite | PASS | exit=0 | 5.66 |
| ruff | PASS | exit=0 | 0.07 |
| compileall | PASS | exit=0 | 0.10 |
| registry verification | PASS | exit=0 | 0.09 |
| research regression suite | SKIP | set EMBEDFLOW_RESEARCH_ROOT to a checkout containing tests/ and tests_mvp/ | 0.00 |
| package build | PASS | exit=0 | 2.84 |
| package artifact audit | PASS |  | 0.00 |
| fresh wheel install | PASS | installed wheel and ran help, doctor, registry verify, and demo | 0.00 |
| FAISS example command 1 | PASS | exit=0 | 0.35 |
| FAISS example command 2 | PASS | exit=0 | 0.32 |
| FAISS example command 3 | PASS | exit=0 | 0.38 |
| CLI registry list | PASS | exit=0 | 0.21 |
| CLI registry show | PASS | exit=0 | 0.21 |
| CLI registry match | PASS | exit=0 | 0.27 |
| CLI registry verify | PASS | exit=0 | 0.20 |
| CLI doctor | PASS | exit=0 | 2.09 |
| CLI search | PASS | exit=0 | 0.32 |
| CLI prewarm | PASS | exit=0 | 0.27 |
| CLI audit-index | PASS | exit=0 | 0.25 |
| CLI economics | PASS | exit=0 | 0.19 |
| CLI export-target | PASS | exit=0 | 0.32 |
| CLI migrate | PASS | exit=0 | 0.35 |
| CLI benchmark profiles | PASS | exit=0 | 0.16 |
| CLI Qdrant demo | PASS | exit=0 | 5.07 |
| real local Qdrant model smoke | SKIP | set EMBEDFLOW_MODEL_ROOT to a directory containing minilm_l6/ and qwen3_0_6b/ snapshots | 0.00 |
| FAISS example server | SKIP | sandbox disallows local sockets | 0.00 |
| research example command 1 | PASS | exit=0 | 0.32 |
| research example command 2 | PASS | exit=0 | 0.28 |
| Qdrant example command 1 | PASS | exit=0 | 1.62 |
| Qdrant example command 2 | PASS | exit=0 | 1.46 |
| Qdrant example server | SKIP | sandbox disallows local sockets | 0.00 |
| documentation links | PASS |  | 0.00 |
| security scan | PASS |  | 0.00 |
| manual metadata audit | PASS | none | 0.00 |
| Python 3.10 smoke | SKIP | interpreter unavailable | 0.00 |
| Python 3.11 smoke | PASS | exit=0 | 1.18 |
| Python 3.12 smoke | PASS | exit=0 | 5.51 |

## Test counts

```json
{
  "passed": 54,
  "skipped": 2
}
```

## Python matrix

- Python 3.10: **SKIP**
- Python 3.11: **PASS**
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

Expected skips are unavailable Python interpreters or optional Qdrant dependencies. No GitHub or external deployment commands are executed by this script.
