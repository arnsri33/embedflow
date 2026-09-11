# Pinecone smoke example

This example points EmbedFlow at an existing Pinecone dense index. It performs
read-only health, stats, and (when a query vector is supplied) retrieval calls;
it does not create an index or write vectors.

Install the optional adapter:

```bash
python -m pip install "embedflow[pinecone]"
```

Set credentials and the data-plane host:

```bash
export PINECONE_API_KEY='...'
export PINECONE_INDEX_HOST='my-index-xxxxx.svc.aped-xxxx.pinecone.io'
./run_smoke.sh
```

For the YAML example, either replace its placeholder host or use the normal
configuration override:

```bash
export EMBEDFLOW_PINECONE_HOST="$PINECONE_INDEX_HOST"
```

The generated configuration uses `api_key_env: PINECONE_API_KEY`; no key is
stored in this directory. For a complete migration, copy
`embedflow.yaml.example` to a working config, provide the source/target model
contracts and either an external JSONL document store or
`text_metadata_field`.
