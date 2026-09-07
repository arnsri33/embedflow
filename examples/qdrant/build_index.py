from pathlib import Path

from embedflow.config import load_config
from embedflow.indexes import QdrantIndex
from embedflow.migration.state import DocumentStore
from embedflow.models import HashEmbeddingModel

root = Path(__file__).resolve().parent
cfg = load_config(root / "embedflow.yaml")
docs = DocumentStore(cfg.documents.path)
model = HashEmbeddingModel("embedflow/demo-source", 64)
vectors = model.encode_documents(list(docs.documents.values()))
index = QdrantIndex.build(cfg.index.path, cfg.index.collection, vectors, list(docs.documents), documents=docs.documents)
index.close()
print(f"built {cfg.index.collection!r} at {cfg.index.path}")
