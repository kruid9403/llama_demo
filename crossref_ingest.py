import hashlib
import os
import re
from html import unescape
from typing import Dict, Iterable, List

import psycopg
import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from vector_store.embeddings import LocalEmbedder

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")
CROSSREF_API = "https://api.crossref.org/works"
TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 400
OVERLAP = 20
DEFAULT_MAX_RESULTS = 10
ALLOWED_TYPES = {"journal-article", "proceedings-article"}

embedder = LocalEmbedder()


def _chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    tokens = TOKENIZER.encode(text)
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        piece = tokens[i : i + max_tokens]
        chunks.append(TOKENIZER.decode(piece))
        i += max_tokens - OVERLAP
    return chunks


def _strip_jats(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(unescape(text), "lxml")
    clean = soup.get_text(" ", strip=True)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def search_crossref(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> Iterable[Dict]:
    params = {
        "query": query,
        "rows": max_results,
        "sort": "relevance",
        "order": "desc",
        "select": "DOI,title,abstract,type,container-title,URL,published-print,published-online,author",
    }
    resp = requests.get(CROSSREF_API, params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("message", {}).get("items", [])

    for item in items:
        item_type = (item.get("type") or "").strip()
        if item_type not in ALLOWED_TYPES:
            continue

        title_list = item.get("title") or []
        title = (title_list[0] if title_list else "").strip()
        abstract = _strip_jats(item.get("abstract") or "")
        doi = (item.get("DOI") or "").strip()
        venue_list = item.get("container-title") or []
        venue = (venue_list[0] if venue_list else "").strip()
        url = (item.get("URL") or "").strip() or (f"https://doi.org/{doi}" if doi else "")

        if not doi or not title:
            continue

        published = item.get("published-print") or item.get("published-online") or {}
        published_parts = published.get("date-parts", [[None]])
        year = published_parts[0][0] if published_parts and published_parts[0] else None

        yield {
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "venue": venue,
            "url": url,
            "year": year,
            "type": item_type,
        }


def _get_or_create_source(cur: psycopg.Cursor) -> int:
    base_url = "https://api.crossref.org"
    cur.execute("SELECT id FROM sources WHERE base_url = %s ORDER BY id ASC LIMIT 1;", (base_url,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO sources (name, base_url, doc_type)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        ("Crossref", base_url, "peer_reviewed_index"),
    )
    return cur.fetchone()[0]


def _paper_text(item: Dict) -> str:
    parts = [
        f"Title: {item['title']}",
        f"Venue: {item['venue']}" if item.get("venue") else "",
        f"Year: {item['year']}" if item.get("year") else "",
        f"DOI: {item['doi']}",
        f"Type: {item['type']}",
        f"Abstract: {item['abstract']}" if item.get("abstract") else "",
    ]
    return "\n".join(p for p in parts if p).strip()


def ingest_crossref_item(item: Dict) -> str | None:
    text = _paper_text(item)
    if not text:
        return None

    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = _chunk_text(text)
    vectors = embedder.embed_passages(chunks)

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            source_id = _get_or_create_source(cur)
            doc_url = item.get("url") or f"https://doi.org/{item['doi']}"

            cur.execute(
                """
                INSERT INTO documents (source_id, url, title, version, checksum)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET checksum = EXCLUDED.checksum
                RETURNING id;
                """,
                (source_id, doc_url, item["title"], item.get("type"), checksum),
            )
            document_id = cur.fetchone()[0]

            cur.execute(
                """
                SELECT id
                FROM sections
                WHERE document_id = %s AND heading = %s AND position = %s
                ORDER BY id ASC
                LIMIT 1;
                """,
                (document_id, "Abstract", 0),
            )
            row = cur.fetchone()
            if row is not None:
                section_id = row[0]
            else:
                cur.execute(
                    """
                    INSERT INTO sections (document_id, heading, level, position)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (document_id, "Abstract", 1, 0),
                )
                section_id = cur.fetchone()[0]

            for chunk, vec in zip(chunks, vectors):
                content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    INSERT INTO chunks (
                        section_id,
                        content,
                        content_hash,
                        content_type,
                        token_count,
                        embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (section_id, content_hash) DO NOTHING
                    RETURNING id;
                    """,
                    (section_id, chunk, content_hash, "abstract", len(chunk.split()), vec),
                )
                cur.fetchone()

    print(f"[ingested-crossref] {item['doi']}")
    return doc_url


def ingest_crossref_query(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> int:
    count, _ = ingest_crossref_query_with_urls(query, max_results=max_results)
    return count


def ingest_crossref_query_with_urls(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> tuple[int, list[str]]:
    ingested = 0
    ingested_urls: list[str] = []
    for item in search_crossref(query, max_results=max_results):
        try:
            doc_url = ingest_crossref_item(item)
            if doc_url:
                ingested += 1
                ingested_urls.append(doc_url)
        except Exception as exc:
            print(f"[crossref-error] {item.get('doi', 'unknown')} -> {exc}")
    return ingested, ingested_urls


if __name__ == "__main__":
    q = input("Enter Crossref query: ").strip()
    if q:
        n = ingest_crossref_query(q)
        print(f"Ingested {n} Crossref records.")
