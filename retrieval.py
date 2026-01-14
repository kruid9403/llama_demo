# retrieval.py
import psycopg
from embeddings import LocalEmbedder

DB_URL = "postgresql://dev_user:dev_password@localhost:5432/embedding_db"
TOP_K = 8

embedder = LocalEmbedder()

def retrieve_chunks(question: str):
    query_vec = embedder.embed_query(question)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.content,
                    c.embedding <=> %s::vector AS distance,
                    d.title,
                    s.heading
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vec, query_vec, TOP_K),
            )
            rows = cur.fetchall()

    return [
        {
            "content": r[0],
            "distance": r[1],
            "document": r[2],
            "section": r[3],
        }
        for r in rows
    ]
