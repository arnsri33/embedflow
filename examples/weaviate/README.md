# Weaviate example

This example uses the current Weaviate Python client (v4) and a local
standalone Weaviate server. Both the HTTP API and gRPC data plane are exposed.
The fixture is externally vectorized and is only used for a local functional
demo; EmbedFlow's production adapter never creates or changes collections.

```bash
cd examples/weaviate
docker compose -f compose.yaml up -d
python -m pip install -e "../../[weaviate,dashboard]"
python ../../scripts/weaviate_fixture.py --uri http://127.0.0.1:8080 --collection Documents --rows 100
cp embedflow.yaml.example embedflow.yaml
python ../../scripts/weaviate_smoke.py --uri http://127.0.0.1:8080 --collection Documents
```

For an authenticated deployment set the environment named by `api_key_env`:

```bash
export WEAVIATE_API_KEY='your-key'
```

Use `text_property` when text is stored in Weaviate. For an external JSONL
document store, keep the same object UUIDs as the document IDs and provide the
document path; candidate retrieval then requests no Weaviate properties.

The source collection is read-only during `doctor`, `audit-index`, `analyze`,
`search`, `serve`, and migration. Stop the disposable server with:

```bash
docker compose -f compose.yaml down -v
```
