# Economics

The economics command estimates the work needed to encode a corpus from a
throughput and GPU price supplied by the user:

```bash
embedflow economics \
  --corpus-size 1000000000 \
  --docs-per-second 100 \
  --gpu-price 3.29
```

The report includes full-backfill GPU hours, one-GPU wall time, estimated cost,
current cache coverage, and remaining work when a cache is supplied.

The calculation is straightforward:

```text
seconds = documents / documents_per_second
gpu_hours = seconds / 3600
estimated_cost = gpu_hours * gpu_price_per_hour
```

Values are projections based on the workload and hardware profile represented
by the inputs. The registry keeps measured throughput profiles separate from
compatibility evidence; `benchmark-profiles list` shows their provenance.
