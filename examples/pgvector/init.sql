-- The demo setup script owns this disposable database only. EmbedFlow's
-- production adapter is read-only and never runs these statements.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.documents (
    id text PRIMARY KEY,
    content text NOT NULL,
    embedding vector(64) NOT NULL
);
