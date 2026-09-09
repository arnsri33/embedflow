"""Create the disposable pgvector fixture used by the local example."""

from __future__ import annotations

import os
import time

from embedflow.models import HashEmbeddingModel

DSN = os.environ.get("EMBEDFLOW_PGVECTOR_DSN", "postgresql://embedflow:embedflow@localhost:5432/embedflow")
TOPICS = [
    ("aurora", "Auroras are caused when charged particles from the solar wind interact with gases in Earth's upper atmosphere."),
    ("coffee", "Coffee beans are roasted seeds whose flavor depends on origin, roast temperature, and brewing method."),
    ("battery", "Lithium-ion batteries store energy through reversible movement of lithium ions between electrodes."),
    ("volcano", "Volcanoes form when magma rises through weaknesses in Earth's crust and erupts at the surface."),
    ("rainbow", "A rainbow appears when sunlight is refracted, reflected, and dispersed by water droplets."),
    ("photosynthesis", "Plants use photosynthesis to convert light, water, and carbon dioxide into chemical energy."),
    ("ocean", "Ocean currents transport heat around the planet and influence climate and marine ecosystems."),
    ("sleep", "Sleep supports memory consolidation, immune function, and recovery from daily activity."),
]


def main() -> None:
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit('Install the optional dependency first: python -m pip install "embedflow[pgvector]"') from exc
    for attempt in range(30):
        try:
            connection = psycopg.connect(DSN)
            break
        except Exception:
            if attempt == 29:
                raise
            time.sleep(1)
    model = HashEmbeddingModel("embedflow/demo-source", 64)
    rows = [(f"doc-{i:03d}", f"{text} Reference note {i} about {topic}.") for i in range(48) for topic, text in [TOPICS[i % len(TOPICS)]]]
    vectors = model.encode_documents([text for _, text in rows])
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("""CREATE TABLE IF NOT EXISTS public.documents (
                id text PRIMARY KEY, content text NOT NULL, embedding vector(64) NOT NULL
            )""")
            cursor.execute("DELETE FROM public.documents")
            for (document_id, text), vector in zip(rows, vectors):
                literal = "[" + ",".join(str(float(value)) for value in vector) + "]"
                cursor.execute("INSERT INTO public.documents (id, content, embedding) VALUES (%s, %s, %s::vector)",
                               (document_id, text, literal))
            cursor.execute("DROP INDEX IF EXISTS documents_embedding_hnsw")
            cursor.execute("CREATE INDEX documents_embedding_hnsw ON public.documents USING hnsw (embedding vector_cosine_ops)")
    connection.close()
    print(f"loaded {len(rows)} documents into public.documents")


if __name__ == "__main__":
    main()
