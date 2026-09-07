# API

The dashboard extra provides a FastAPI application. Start it with:

```bash
embedflow serve --config ./embedflow.yaml --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at
<http://127.0.0.1:8000/docs>.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Process and backend health |
| GET | `/status` | Cache, queue, model, and migration state |
| POST | `/search` | Source retrieval and target reranking |
| POST | `/analyze` | Run or retrieve migration analysis |
| POST | `/prewarm` | Queue document materialization |
| GET | `/metrics` | Aggregated latency and queue metrics |
| GET | `/plan` | Current migration plan |
| GET | `/economics` | Configured economics estimate |

## Search

Request:

```json
{"query": "what causes auroras?", "top_k": 10}
```

The response includes the result list and migration fields such as:

```json
{
  "state": "PARTIAL",
  "candidate_depth": 50,
  "cache_hits": 42,
  "cache_misses": 8,
  "sync_encoded": 4,
  "async_queued": 4,
  "results": []
}
```

`COLD`, `PARTIAL`, and `WARM` describe target-vector availability for that
request. A partial response scores the available target vectors; it can differ
from the fully warm ranking.

## Errors

Validation errors use FastAPI/Pydantic's normal JSON response. Backend,
encoder, and cache failures are returned as structured errors and are also
recorded in the configured telemetry log.
