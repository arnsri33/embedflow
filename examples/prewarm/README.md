# Traffic-aware prewarm demo

This offline demo creates a tiny deterministic Shadow telemetry store, warms a
small part of the target cache, generates a traffic-hotset plan, and executes
that bounded plan through EmbedFlow's existing materialization worker.

```bash
cd examples/prewarm
./run_demo.sh
```

The generated `runtime/` directory is disposable and is not part of the
package. The percentages describe observed candidate-occurrence coverage in
the synthetic window only; they are not retrieval-quality or recall claims.
For a starting configuration shape, see `embedflow.yaml.example`; production
plans must still point at an existing source index and its current telemetry.
