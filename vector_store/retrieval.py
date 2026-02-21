# retrieval.py
import os

import psycopg
from dotenv import load_dotenv

from vector_store.embeddings import LocalEmbedder

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")
TOP_K = 8
RECENT_DOCS_LIMIT = int(os.getenv("RECENT_DOCS_LIMIT", "10"))

embedder = LocalEmbedder()

def retrieve_chunks(
    question: str,
    document_urls: list[str] | None = None,
    top_k: int = TOP_K,
    recent_only: bool = True,
):
    query_vec = embedder.embed_query(question)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            if document_urls:
                cur.execute(
                    """
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    WHERE d.url = ANY(%s)
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vec, document_urls, query_vec, top_k),
                )
            elif recent_only:
                cur.execute(
                    """
                    WITH recent_docs AS (
                        SELECT id
                        FROM documents
                        ORDER BY created_at DESC
                        LIMIT %s
                    )
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    JOIN recent_docs r ON r.id = d.id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (RECENT_DOCS_LIMIT, query_vec, query_vec, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vec, query_vec, top_k),
                )
            rows = cur.fetchall()

    return [
        {
            "content": r[0],
            "distance": r[1],
            "document": r[2],
            "section": r[3],
            "url": r[4],
        }
        for r in rows
    ]
