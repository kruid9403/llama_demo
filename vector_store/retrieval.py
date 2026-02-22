# retrieval.py
import os

import psycopg
from dotenv import load_dotenv

from vector_store.embeddings import LocalEmbedder

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")
TOP_K = 8
RECENT_DOCS_LIMIT = int(os.getenv("RECENT_DOCS_LIMIT", "10"))
CITATION_WEIGHT_BOOST = float(os.getenv("CITATION_WEIGHT_BOOST", "0.03"))
CITATION_WEIGHT_MAX_BONUS = float(os.getenv("CITATION_WEIGHT_MAX_BONUS", "0.20"))

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
                    WITH citation_counts AS (
                        SELECT
                            lower(doi) AS doi_key,
                            COUNT(DISTINCT document_id)::int AS citation_count
                        FROM document_references
                        WHERE source_url IS NULL
                          AND doi IS NOT NULL
                          AND btrim(doi) <> ''
                        GROUP BY lower(doi)
                    )
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url,
                        COALESCE(cc.citation_count, 0) AS citation_count,
                        (
                            (c.embedding <=> %s::vector)
                            - LEAST(
                                %s::double precision,
                                LN(1 + COALESCE(cc.citation_count, 0)) * %s::double precision
                            )
                        ) AS weighted_distance
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    LEFT JOIN LATERAL (
                        SELECT doi
                        FROM document_references
                        WHERE document_id = d.id
                        ORDER BY (source_url IS NOT NULL) DESC, id DESC
                        LIMIT 1
                    ) dr ON TRUE
                    LEFT JOIN citation_counts cc ON cc.doi_key = lower(dr.doi)
                    WHERE d.url = ANY(%s)
                    ORDER BY weighted_distance, distance
                    LIMIT %s;
                    """,
                    (
                        query_vec,
                        query_vec,
                        CITATION_WEIGHT_MAX_BONUS,
                        CITATION_WEIGHT_BOOST,
                        document_urls,
                        top_k,
                    ),
                )
            elif recent_only:
                cur.execute(
                    """
                    WITH recent_docs AS (
                        SELECT id
                        FROM documents
                        ORDER BY created_at DESC
                        LIMIT %s
                    ),
                    citation_counts AS (
                        SELECT
                            lower(doi) AS doi_key,
                            COUNT(DISTINCT document_id)::int AS citation_count
                        FROM document_references
                        WHERE source_url IS NULL
                          AND doi IS NOT NULL
                          AND btrim(doi) <> ''
                        GROUP BY lower(doi)
                    )
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url,
                        COALESCE(cc.citation_count, 0) AS citation_count,
                        (
                            (c.embedding <=> %s::vector)
                            - LEAST(
                                %s::double precision,
                                LN(1 + COALESCE(cc.citation_count, 0)) * %s::double precision
                            )
                        ) AS weighted_distance
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    JOIN recent_docs r ON r.id = d.id
                    LEFT JOIN LATERAL (
                        SELECT doi
                        FROM document_references
                        WHERE document_id = d.id
                        ORDER BY (source_url IS NOT NULL) DESC, id DESC
                        LIMIT 1
                    ) dr ON TRUE
                    LEFT JOIN citation_counts cc ON cc.doi_key = lower(dr.doi)
                    ORDER BY weighted_distance, distance
                    LIMIT %s;
                    """,
                    (
                        RECENT_DOCS_LIMIT,
                        query_vec,
                        query_vec,
                        CITATION_WEIGHT_MAX_BONUS,
                        CITATION_WEIGHT_BOOST,
                        top_k,
                    ),
                )
            else:
                cur.execute(
                    """
                    WITH citation_counts AS (
                        SELECT
                            lower(doi) AS doi_key,
                            COUNT(DISTINCT document_id)::int AS citation_count
                        FROM document_references
                        WHERE source_url IS NULL
                          AND doi IS NOT NULL
                          AND btrim(doi) <> ''
                        GROUP BY lower(doi)
                    )
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url,
                        COALESCE(cc.citation_count, 0) AS citation_count,
                        (
                            (c.embedding <=> %s::vector)
                            - LEAST(
                                %s::double precision,
                                LN(1 + COALESCE(cc.citation_count, 0)) * %s::double precision
                            )
                        ) AS weighted_distance
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    LEFT JOIN LATERAL (
                        SELECT doi
                        FROM document_references
                        WHERE document_id = d.id
                        ORDER BY (source_url IS NOT NULL) DESC, id DESC
                        LIMIT 1
                    ) dr ON TRUE
                    LEFT JOIN citation_counts cc ON cc.doi_key = lower(dr.doi)
                    ORDER BY weighted_distance, distance
                    LIMIT %s;
                    """,
                    (
                        query_vec,
                        query_vec,
                        CITATION_WEIGHT_MAX_BONUS,
                        CITATION_WEIGHT_BOOST,
                        top_k,
                    ),
                )
            rows = cur.fetchall()

    return [
        {
            "content": r[0],
            "distance": r[1],
            "document": r[2],
            "section": r[3],
            "url": r[4],
            "citation_count": r[5],
            "weighted_distance": r[6],
        }
        for r in rows
    ]


def similarity_search_with_references(
    question: str,
    document_urls: list[str] | None = None,
    top_k: int = TOP_K,
    recent_only: bool = False,
):
    query_vec = embedder.embed_query(question)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            if document_urls:
                cur.execute(
                    """
                    WITH citation_counts AS (
                        SELECT
                            lower(doi) AS doi_key,
                            COUNT(DISTINCT document_id)::int AS citation_count
                        FROM document_references
                        WHERE source_url IS NULL
                          AND doi IS NOT NULL
                          AND btrim(doi) <> ''
                        GROUP BY lower(doi)
                    )
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url,
                        dr.source_url,
                        dr.doi,
                        dr.venue,
                        dr.published_date,
                        dr.authors,
                        COALESCE(cc.citation_count, 0) AS citation_count,
                        (
                            (c.embedding <=> %s::vector)
                            - LEAST(
                                %s::double precision,
                                LN(1 + COALESCE(cc.citation_count, 0)) * %s::double precision
                            )
                        ) AS weighted_distance
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    LEFT JOIN LATERAL (
                        SELECT source_url, doi, venue, published_date, authors
                        FROM document_references
                        WHERE document_id = d.id
                        ORDER BY (source_url IS NOT NULL) DESC, id DESC
                        LIMIT 1
                    ) dr ON TRUE
                    LEFT JOIN citation_counts cc ON cc.doi_key = lower(dr.doi)
                    WHERE d.url = ANY(%s)
                    ORDER BY weighted_distance, distance
                    LIMIT %s;
                    """,
                    (
                        query_vec,
                        query_vec,
                        CITATION_WEIGHT_MAX_BONUS,
                        CITATION_WEIGHT_BOOST,
                        document_urls,
                        top_k,
                    ),
                )
            elif recent_only:
                cur.execute(
                    """
                    WITH recent_docs AS (
                        SELECT id
                        FROM documents
                        ORDER BY created_at DESC
                        LIMIT %s
                    ),
                    citation_counts AS (
                        SELECT
                            lower(doi) AS doi_key,
                            COUNT(DISTINCT document_id)::int AS citation_count
                        FROM document_references
                        WHERE source_url IS NULL
                          AND doi IS NOT NULL
                          AND btrim(doi) <> ''
                        GROUP BY lower(doi)
                    )
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url,
                        dr.source_url,
                        dr.doi,
                        dr.venue,
                        dr.published_date,
                        dr.authors,
                        COALESCE(cc.citation_count, 0) AS citation_count,
                        (
                            (c.embedding <=> %s::vector)
                            - LEAST(
                                %s::double precision,
                                LN(1 + COALESCE(cc.citation_count, 0)) * %s::double precision
                            )
                        ) AS weighted_distance
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    JOIN recent_docs r ON r.id = d.id
                    LEFT JOIN LATERAL (
                        SELECT source_url, doi, venue, published_date, authors
                        FROM document_references
                        WHERE document_id = d.id
                        ORDER BY (source_url IS NOT NULL) DESC, id DESC
                        LIMIT 1
                    ) dr ON TRUE
                    LEFT JOIN citation_counts cc ON cc.doi_key = lower(dr.doi)
                    ORDER BY weighted_distance, distance
                    LIMIT %s;
                    """,
                    (
                        RECENT_DOCS_LIMIT,
                        query_vec,
                        query_vec,
                        CITATION_WEIGHT_MAX_BONUS,
                        CITATION_WEIGHT_BOOST,
                        top_k,
                    ),
                )
            else:
                cur.execute(
                    """
                    WITH citation_counts AS (
                        SELECT
                            lower(doi) AS doi_key,
                            COUNT(DISTINCT document_id)::int AS citation_count
                        FROM document_references
                        WHERE source_url IS NULL
                          AND doi IS NOT NULL
                          AND btrim(doi) <> ''
                        GROUP BY lower(doi)
                    )
                    SELECT
                        c.content,
                        c.embedding <=> %s::vector AS distance,
                        d.title,
                        s.heading,
                        d.url,
                        dr.source_url,
                        dr.doi,
                        dr.venue,
                        dr.published_date,
                        dr.authors,
                        COALESCE(cc.citation_count, 0) AS citation_count,
                        (
                            (c.embedding <=> %s::vector)
                            - LEAST(
                                %s::double precision,
                                LN(1 + COALESCE(cc.citation_count, 0)) * %s::double precision
                            )
                        ) AS weighted_distance
                    FROM chunks c
                    JOIN sections s ON s.id = c.section_id
                    JOIN documents d ON d.id = s.document_id
                    LEFT JOIN LATERAL (
                        SELECT source_url, doi, venue, published_date, authors
                        FROM document_references
                        WHERE document_id = d.id
                        ORDER BY (source_url IS NOT NULL) DESC, id DESC
                        LIMIT 1
                    ) dr ON TRUE
                    LEFT JOIN citation_counts cc ON cc.doi_key = lower(dr.doi)
                    ORDER BY weighted_distance, distance
                    LIMIT %s;
                    """,
                    (
                        query_vec,
                        query_vec,
                        CITATION_WEIGHT_MAX_BONUS,
                        CITATION_WEIGHT_BOOST,
                        top_k,
                    ),
                )
            rows = cur.fetchall()

    return [
        {
            "content": r[0],
            "distance": r[1],
            "document": r[2],
            "section": r[3],
            "url": r[4],
            "source_url": r[5],
            "doi": r[6],
            "venue": r[7],
            "published_date": r[8],
            "authors": r[9],
            "citation_count": r[10],
            "weighted_distance": r[11],
        }
        for r in rows
    ]
