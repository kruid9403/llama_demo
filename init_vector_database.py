import os
from typing import List
from contextlib import contextmanager
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
import psycopg
import requests
from bs4 import BeautifulSoup
import numpy as np
import hashlib
import tiktoken

DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5432/embedding_db")

EMBEDDING_DIM = 384
CHUNK_SIZE = 500

TOKENIZER = tiktoken.get_encoding("cl100k_base") 
MAX_TOKENS = 400 
OVERLAP = 50

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def chunk_tokens(text: str) -> List[str]:
    tokens = TOKENIZER.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + MAX_TOKENS]
        chunks.append(TOKENIZER.decode(chunk))
        i += MAX_TOKENS - OVERLAP

    return chunks

#----------- Embedding function -----------

def embed_texts(texts: List[str]) -> List[List[float]]:
    return[np.random.rand(EMBEDDING_DIM).tolist() for _ in texts]


#----------- Chunking function -----------

def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]

#----------- Main Insert Function -----------
def add_vectors_from_url():
    url = input("Enter URL to add vectors from: ").strip()
    print(f"Scraping: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    clean_text = soup.get_text(separator=" ", strip=True)

    if not clean_text:
        print("No extractable text found.")
        return

    checksum = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    chunks = chunk_text(clean_text, CHUNK_SIZE)
    embeddings = embed_texts(chunks)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            # 1️⃣ Insert source
            cur.execute("""
                INSERT INTO sources (name, base_url, doc_type)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING id;
            """, ("Documentation Site", url, "docs"))

            source_id = cur.fetchone()
            if source_id is None:
                cur.execute(
                    "SELECT id FROM sources WHERE base_url = %s;",
                    (url,)
                )
                source_id = cur.fetchone()

            source_id = source_id[0]

            # 2️⃣ Insert document
            cur.execute("""
                INSERT INTO documents (source_id, url, checksum)
                VALUES (%s, %s, %s)
                ON CONFLICT (url) DO NOTHING
                RETURNING id;
            """, (source_id, url, checksum))

            doc_id = cur.fetchone()
            if doc_id is None:
                cur.execute(
                    "SELECT id FROM documents WHERE url = %s;",
                    (url,)
                )
                doc_id = cur.fetchone()

            document_id = doc_id[0]

            # 3️⃣ Insert single section (flat scrape)
            cur.execute("""
                INSERT INTO sections (document_id, heading, level, position)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (document_id, "Main Content", 1, 0))

            section_id = cur.fetchone()[0]

            chunk_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

            cur.execute("""
                INSERT INTO chunks (
                    section_id,
                    content,
                    content_hash,
                    content_type,
                    token_count,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (section_id, content_hash) DO NOTHING;
            """, (
                section_id,
                clean_text,
                chunk_hash,
                "prose",
                count_tokens(clean_text),
                embeddings
            ))


    print(f"Inserted {len(chunks)} chunks into pgvector.")

def init_vector_database():
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            # Enable pgvector
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    doc_type TEXT,
                    created_at TIMESTAMP DEFAULT now()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    version TEXT,
                    last_scraped TIMESTAMP,
                    checksum TEXT,
                    created_at TIMESTAMP DEFAULT now()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS sections (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    heading TEXT,
                    level INTEGER,
                    anchor TEXT,
                    position INTEGER
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    content_type TEXT,
                    language TEXT,
                    token_count INTEGER,
                    created_at TIMESTAMP DEFAULT now()
                );
            """)

            # 🔑 REQUIRED MIGRATION
            cur.execute("""
                ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS embedding VECTOR(384);
            """)

            # Vector index (safe now)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops);
            """)

    print("Vector schema initialized successfully.")


if __name__ == "__main__":
    init_vector_database()
    add_vectors_from_url()
    