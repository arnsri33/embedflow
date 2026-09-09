# pgvector migration example

This example starts a disposable PostgreSQL 16 database with pgvector, loads
48 deterministic demo documents into an existing `public.documents` table,
and runs the regular EmbedFlow analysis and serving commands against it.

Requirements: Docker Compose, Python 3.10+, and the optional adapter:

```bash
python -m pip install "embedflow[pgvector,dashboard]"
docker compose up -d
./run_demo.sh
```

Open <http://127.0.0.1:8000/> and search for `what causes auroras?`. The
database credentials above are for the disposable local container only. The
production adapter reads `EMBEDFLOW_PGVECTOR_DSN` and performs retrieval/text
lookups without creating or modifying a user's schema or index.

Stop and remove the demo database with:

```bash
docker compose down -v
```

To use a real table, copy the `index` section from `embedflow.yaml`, set your
own DSN environment variable, and follow
[`docs/integrations/pgvector.md`](../../docs/integrations/pgvector.md).
