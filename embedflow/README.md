# EmbedFlow package

This directory contains the installable Python package. Start with the
[repository README](../README.md) for the architecture, quickstart, CLI/API
examples, Qdrant configuration, compatibility wording, limitations, and demo
instructions.

The package reuses the frozen embedding contracts and `src.t2_v1` decision
code. It keeps a legacy FAISS/Qdrant index online, target-reranks a bounded
candidate set, persists target vectors in SQLite, and materializes misses
through a retrying background worker.
