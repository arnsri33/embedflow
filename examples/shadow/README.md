# Shadow Mode demo

This network-free demo uses deterministic hash encoders and a tiny in-memory
source index. It demonstrates that the source result is returned immediately,
shadow work is scheduled in the background, and the target cache/materializer
eventually records an observation.

```bash
./run_demo.sh
```

The demo is intentionally synthetic. Its report is operational diagnostics,
not retrieval-quality evidence.
