# EmbedFlow v0.1.0 Release Test Report

- Generated: 2026-09-07 13:19:08 -0700
- Python running gate: 3.12.12
- Repository: `embedflow` (local release checkout)
- Git commit: uncommitted

## Gate

**RELEASE GATE: PASS**

## Check matrix

| Check | Status | Detail | Seconds |
|---|---|---|---:|
| public pytest suite | PASS | exit=0 | 4.23 |
| ruff | PASS | exit=0 | 0.10 |
| compileall | PASS | exit=0 | 0.08 |
| registry verification | PASS | exit=0 | 0.09 |
| research regression suite | PASS | exit=0 | 6.99 |
| package build | PASS | exit=0 | 2.78 |
| package artifact audit | PASS |  | 0.00 |
| fresh wheel install | PASS | installed wheel and ran help, doctor, registry verify, and demo | 0.00 |
| FAISS example command 1 | PASS | exit=0 | 0.32 |
| FAISS example command 2 | PASS | exit=0 | 0.33 |
| FAISS example command 3 | PASS | exit=0 | 0.31 |
| CLI registry list | PASS | exit=0 | 0.22 |
| CLI registry show | PASS | exit=0 | 0.24 |
| CLI registry match | PASS | exit=0 | 0.23 |
| CLI registry verify | PASS | exit=0 | 0.19 |
| CLI doctor | PASS | exit=0 | 2.01 |
| CLI search | PASS | exit=0 | 0.34 |
| CLI prewarm | PASS | exit=0 | 0.27 |
| CLI audit-index | PASS | exit=0 | 0.28 |
| CLI economics | PASS | exit=0 | 0.21 |
| CLI export-target | PASS | exit=0 | 0.35 |
| CLI migrate | PASS | exit=0 | 0.40 |
| CLI benchmark profiles | PASS | exit=0 | 0.20 |
| CLI Qdrant demo | PASS | exit=0 | 5.01 |
| real local Qdrant model smoke | PASS | exit=0 | 17.08 |
| FAISS example server | SKIP | sandbox disallows local sockets | 0.00 |
| research example command 1 | PASS | exit=0 | 0.36 |
| research example command 2 | PASS | exit=0 | 0.31 |
| Qdrant example command 1 | PASS | exit=0 | 1.49 |
| Qdrant example command 2 | PASS | exit=0 | 1.52 |
| Qdrant example server | SKIP | sandbox disallows local sockets | 0.00 |
| documentation links | PASS |  | 0.00 |
| security scan | PASS |  | 0.00 |
| manual metadata audit | PASS | none | 0.00 |
| Python 3.10 smoke | SKIP | interpreter unavailable | 0.00 |
| Python 3.11 smoke | SKIP | pytest is not installed in this interpreter | 0.00 |
| Python 3.12 smoke | PASS | exit=0 | 3.46 |

## Test counts

```json
{
  "passed": 54
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
Retained artifacts verified: 21
Semantic values cross-checked: 595
Provenance and packaged checksums are validated by `registry verify`.

## Security

The source scan checks credentials, private keys, absolute local paths, and runtime artifacts. Manual metadata placeholders are a separate release blocker.

## Notes

Expected skips are unavailable Python interpreters or optional Qdrant dependencies. No GitHub or external deployment commands are executed by this script.
