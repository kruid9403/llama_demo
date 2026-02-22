import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import psycopg
import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from vector_store.embeddings import EMBEDDING_DIM, LocalEmbedder

load_dotenv()

DB_URL = os.getenv("DB_URL", "postgresql://dev_user:dev_password@localhost:5433/embedding_db")
TOKENIZER: tiktoken.Encoding | None = None
TOKENIZER_FAILED = False
MAX_TOKENS = int(os.getenv("SPIDER_MAX_TOKENS", "400"))
OVERLAP = int(os.getenv("SPIDER_CHUNK_OVERLAP", "50"))
USER_AGENT = os.getenv("SPIDER_USER_AGENT", "llama-demo-scholar-spider/1.0")
DEFAULT_SEED_URL = os.getenv("SPIDER_DEFAULT_SEED_URL", "https://arxiv.org")
DEFAULT_ALLOWED_DOMAINS = tuple(
    d.strip().lower()
    for d in os.getenv("SCHOLAR_ALLOWED_DOMAINS", "arxiv.org").split(",")
    if d.strip()
)
MIN_CONTENT_CHARS = int(os.getenv("SPIDER_MIN_CONTENT_CHARS", "500"))
MAX_CITED_DOIS = int(os.getenv("SPIDER_MAX_CITED_DOIS", "200"))
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
SKIP_FILE_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".zip",
    ".tar",
    ".gz",
    ".mp4",
    ".mp3",
    ".svg",
}


SpiderEventCallback = Callable[[dict], None]


@dataclass
class CrawledDocument:
    url: str
    title: str
    abstract: str
    body: str
    doi: str
    venue: str
    published_date: str
    authors: list[str]
    domain: str


def _emit_event(
    callback: SpiderEventCallback | None,
    event_type: str,
    message: str,
    **payload,
) -> None:
    event = {"type": event_type, "message": message, **payload}
    if callback is None:
        print(message)
        return
    callback(event)


class RobotsCache:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._cache: dict[str, RobotFileParser | None] = {}

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False

        site_root = f"{parsed.scheme}://{parsed.netloc}"
        robot = self._cache.get(site_root)
        if robot is None and site_root not in self._cache:
            rp = RobotFileParser()
            rp.set_url(f"{site_root}/robots.txt")
            try:
                rp.read()
                self._cache[site_root] = rp
            except Exception:
                self._cache[site_root] = None
            robot = self._cache[site_root]

        if robot is None:
            return True
        try:
            return robot.can_fetch(self.user_agent, url)
        except Exception:
            return True


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _get_tokenizer() -> tiktoken.Encoding | None:
    global TOKENIZER, TOKENIZER_FAILED
    if TOKENIZER is not None:
        return TOKENIZER
    if TOKENIZER_FAILED:
        return None
    try:
        TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        print(f"[tokenizer-warning] Falling back to word-based chunking: {exc}")
        TOKENIZER_FAILED = True
        TOKENIZER = None
    return TOKENIZER


def _count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text))


def _chunk_text(text: str) -> list[str]:
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        i = 0
        word_chunk = max(1, int(MAX_TOKENS * 0.75))
        word_overlap = max(0, int(OVERLAP * 0.75))
        while i < len(words):
            piece = words[i : i + word_chunk]
            chunks.append(" ".join(piece))
            step = max(1, word_chunk - word_overlap)
            i += step
        return chunks

    tokens = tokenizer.encode(text)
    chunks: list[str] = []
    i = 0
    while i < len(tokens):
        piece = tokens[i : i + MAX_TOKENS]
        chunks.append(tokenizer.decode(piece))
        i += MAX_TOKENS - OVERLAP
    return chunks


def _canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    if base_url:
        url = urljoin(base_url, url)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    netloc = parsed.netloc.lower()
    if netloc.endswith(":80") and parsed.scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and parsed.scheme == "https":
        netloc = netloc[:-4]

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    cleaned = parsed._replace(netloc=netloc, path=path, fragment="")
    return urlunparse((cleaned.scheme, cleaned.netloc, cleaned.path, "", cleaned.query, ""))


def _is_allowed_domain(url: str, allowed_domains: Iterable[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith(f".{d}") for d in allowed_domains)


def _is_skipped_file_url(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return any(path.endswith(ext) for ext in SKIP_FILE_EXTENSIONS)


def _is_author_based_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        path = (parsed.path or "").lower()
        if path.startswith("/a/"):
            return True
        if path.startswith("/search"):
            query = parsed.query or ""
            if "searchtype=author" in query.lower():
                return True
            try:
                qs = parse_qs(query, keep_blank_values=True)
            except Exception:
                qs = {}
            searchtypes = [str(v).lower() for v in (qs.get("searchtype") or [])]
            if any(st.startswith("author") for st in searchtypes):
                return True
            for key, values in qs.items():
                key_l = str(key).lower()
                if key_l.startswith("terms-") and key_l.endswith("-field"):
                    if any(str(v).lower() == "author" for v in values):
                        return True
    return False


def _first_meta_content(soup: BeautifulSoup, *keys: str) -> str:
    for key in keys:
        tag = soup.find("meta", attrs={"name": key}) or soup.find("meta", attrs={"property": key})
        if tag and tag.get("content"):
            return _normalize_space(tag["content"])
    return ""


def _collect_meta_values(soup: BeautifulSoup, key: str) -> list[str]:
    values: list[str] = []
    for tag in soup.find_all("meta", attrs={"name": key}):
        content = _normalize_space(tag.get("content", ""))
        if content:
            values.append(content)
    return values


def _extract_cited_dois(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in DOI_PATTERN.finditer(text):
        doi = _normalize_space(match.group(0)).rstrip(".,;:)]}")
        if not doi:
            continue
        doi_l = doi.lower()
        if doi_l in seen:
            continue
        seen.add(doi_l)
        ordered.append(doi)
        if len(ordered) >= MAX_CITED_DOIS:
            break
    return ordered


def _extract_body_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg"]):
        tag.decompose()

    text_candidates: list[str] = []
    article = soup.find("article")
    node = article if article is not None else soup.body
    if node is not None:
        for part in node.find_all(["p", "li"]):
            txt = _normalize_space(part.get_text(" ", strip=True))
            if len(txt) >= 30:
                text_candidates.append(txt)

    body = "\n".join(text_candidates).strip()
    if len(body) < MIN_CONTENT_CHARS:
        body = _normalize_space(soup.get_text(" ", strip=True))
    return body


def _extract_document(url: str, html: str) -> tuple[CrawledDocument | None, BeautifulSoup]:
    soup = BeautifulSoup(html, "lxml")

    title = _first_meta_content(soup, "citation_title", "og:title")
    if not title:
        title = _normalize_space(soup.title.get_text(" ", strip=True) if soup.title else "")

    abstract = _first_meta_content(
        soup,
        "citation_abstract",
        "description",
        "dc.description",
        "dcterms.abstract",
    )
    venue = _first_meta_content(soup, "citation_journal_title", "citation_conference_title", "dc.source")
    published = _first_meta_content(
        soup,
        "citation_publication_date",
        "citation_date",
        "article:published_time",
        "dc.date",
    )
    authors = _collect_meta_values(soup, "citation_author")
    if not authors:
        authors = _collect_meta_values(soup, "dc.creator")

    doi = _first_meta_content(soup, "citation_doi", "dc.identifier")
    if doi:
        doi_match = DOI_PATTERN.search(doi)
        doi = doi_match.group(0) if doi_match else doi
    if not doi:
        doi_from_text = DOI_PATTERN.search(html)
        doi = doi_from_text.group(0) if doi_from_text else ""

    body = _extract_body_text(soup)
    if len(body) < MIN_CONTENT_CHARS:
        return None, soup

    domain = (urlparse(url).hostname or "").lower()
    doc = CrawledDocument(
        url=url,
        title=title or "Untitled scholarly page",
        abstract=abstract,
        body=body,
        doi=doi,
        venue=venue,
        published_date=published,
        authors=authors[:12],
        domain=domain,
    )
    return doc, soup


def _extract_links(soup: BeautifulSoup, base_url: str, allowed_domains: Iterable[str]) -> list[str]:
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        normalized = _canonicalize_url(href, base_url=base_url)
        if not normalized:
            continue
        if _is_skipped_file_url(normalized):
            continue
        if not _is_allowed_domain(normalized, allowed_domains):
            continue
        links.append(normalized)
    return links


def _build_document_text(doc: CrawledDocument) -> str:
    authors = ", ".join(doc.authors) if doc.authors else ""
    parts = [
        f"Title: {doc.title}",
        f"URL: {doc.url}",
        f"DOI: {doc.doi}" if doc.doi else "",
        f"Venue: {doc.venue}" if doc.venue else "",
        f"Published: {doc.published_date}" if doc.published_date else "",
        f"Authors: {authors}" if authors else "",
        f"Abstract: {doc.abstract}" if doc.abstract else "",
        f"Body: {doc.body}",
    ]
    return "\n".join(p for p in parts if p).strip()


def _get_or_create_source(cur: psycopg.Cursor, domain: str) -> int:
    base_url = f"https://{domain}"
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
        (f"Scholarly Spider ({domain})", base_url, "peer_reviewed_crawl"),
    )
    return cur.fetchone()[0]


def _get_or_create_main_section(cur: psycopg.Cursor, document_id: int) -> int:
    cur.execute(
        """
        SELECT id
        FROM sections
        WHERE document_id = %s AND heading = %s AND position = %s
        ORDER BY id ASC
        LIMIT 1;
        """,
        (document_id, "Main Content", 0),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO sections (document_id, heading, level, position)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """,
        (document_id, "Main Content", 1, 0),
    )
    return cur.fetchone()[0]


def _upsert_reference_metadata(cur: psycopg.Cursor, document_id: int, doc: CrawledDocument) -> None:
    cur.execute("DELETE FROM document_references WHERE document_id = %s;", (document_id,))
    cited_dois = _extract_cited_dois(doc.body)
    self_doi = (doc.doi or "").strip().lower()

    for cited_doi in cited_dois:
        cited_doi_l = cited_doi.lower()
        if self_doi and cited_doi_l == self_doi:
            continue
        cur.execute(
            """
            INSERT INTO document_references (
                document_id,
                source_url,
                doi,
                venue,
                published_date,
                authors
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (document_id, None, cited_doi, None, None, None),
        )

    cur.execute(
        """
        INSERT INTO document_references (
            document_id,
            source_url,
            doi,
            venue,
            published_date,
            authors
        )
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (
            document_id,
            doc.url,
            doc.doi or None,
            doc.venue or None,
            doc.published_date or None,
            ", ".join(doc.authors) if doc.authors else None,
        ),
    )


def _ingest_document(cur: psycopg.Cursor, embedder: LocalEmbedder, doc: CrawledDocument) -> int:
    text = _build_document_text(doc)
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()

    source_id = _get_or_create_source(cur, doc.domain)
    cur.execute(
        """
        INSERT INTO documents (source_id, url, title, checksum, last_scraped)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (url) DO UPDATE SET
            title = EXCLUDED.title,
            checksum = EXCLUDED.checksum,
            last_scraped = now()
        RETURNING id;
        """,
        (source_id, doc.url, doc.title, checksum),
    )
    document_id = cur.fetchone()[0]
    section_id = _get_or_create_main_section(cur, document_id)
    _upsert_reference_metadata(cur, document_id, doc)

    chunks = _chunk_text(text)
    vectors = embedder.embed_passages(chunks)
    inserted = 0
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
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (section_id, content_hash) DO NOTHING
            RETURNING id;
            """,
            (section_id, chunk, content_hash, "scholarly_prose", _count_tokens(chunk), vec),
        )
        if cur.fetchone():
            inserted += 1
    return inserted


def _write_raw_snapshot(raw_dir: Path, doc: CrawledDocument, html: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = hashlib.sha1(doc.url.encode("utf-8")).hexdigest()
    html_path = raw_dir / f"{base}.html"
    meta_path = raw_dir / f"{base}.json"

    html_path.write_text(html, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "url": doc.url,
                "title": doc.title,
                "doi": doc.doi,
                "venue": doc.venue,
                "published_date": doc.published_date,
                "authors": doc.authors,
                "domain": doc.domain,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def _ensure_schema(cur: psycopg.Cursor) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            doc_type TEXT,
            created_at TIMESTAMP DEFAULT now()
        );
        """
    )
    cur.execute(
        """
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
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sections (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            heading TEXT,
            level INTEGER,
            anchor TEXT,
            position INTEGER
        );
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            content_hash TEXT,
            content_type TEXT,
            language TEXT,
            token_count INTEGER,
            embedding VECTOR({EMBEDDING_DIM}),
            created_at TIMESTAMP DEFAULT now()
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS document_references (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            source_url TEXT,
            doi TEXT,
            venue TEXT,
            published_date TEXT,
            authors TEXT,
            created_at TIMESTAMP DEFAULT now()
        );
        """
    )
    cur.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;")
    cur.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding VECTOR({EMBEDDING_DIM});")
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS chunks_section_content_hash_uidx
        ON chunks (section_id, content_hash);
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks
        USING hnsw (embedding vector_cosine_ops);
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS document_references_document_source_doi_uidx
        ON document_references (
            document_id,
            COALESCE(source_url, ''),
            COALESCE(doi, '')
        );
        """
    )


def crawl_and_ingest(
    seed_urls: list[str],
    allowed_domains: list[str],
    max_pages: int,
    max_depth: int,
    delay_seconds: float,
    respect_robots: bool,
    raw_dir: Path | None,
    event_callback: SpiderEventCallback | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[int, int]:
    queue: deque[tuple[str, int]] = deque()
    queued: set[str] = set()
    visited: set[str] = set()

    for seed in seed_urls:
        canonical = _canonicalize_url(seed)
        if not canonical:
            continue
        if canonical not in queued:
            queue.append((canonical, 0))
            queued.add(canonical)

    if not queue:
        raise ValueError("No valid seed URLs were provided.")
    max_pages_label = "unlimited" if max_pages <= 0 else str(max_pages)
    max_depth_label = "unlimited" if max_depth < 0 else str(max_depth)
    _emit_event(
        event_callback,
        "start",
        (
            f"[start] seeds={len(queue)} domains={','.join(allowed_domains)} "
            f"max_pages={max_pages_label} max_depth={max_depth_label}"
        ),
        seeds=len(queue),
        domains=allowed_domains,
        max_pages=max_pages,
        max_depth=max_depth,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    robots = RobotsCache(USER_AGENT)
    embedder = LocalEmbedder()

    pages_ingested = 0
    chunk_count = 0

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            _ensure_schema(cur)
            conn.commit()
            while queue and (max_pages <= 0 or pages_ingested < max_pages):
                if stop_event is not None and stop_event.is_set():
                    _emit_event(
                        event_callback,
                        "interrupted",
                        "[interrupted] Crawl stopped by user request.",
                    )
                    break

                url, depth = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)

                if not _is_allowed_domain(url, allowed_domains):
                    _emit_event(event_callback, "skip_domain", f"[skip-domain] {url}", url=url)
                    continue

                if _is_author_based_url(url):
                    _emit_event(event_callback, "skip_author", f"[skip-author] {url}", url=url)
                    continue

                if respect_robots and not robots.can_fetch(url):
                    _emit_event(event_callback, "robots_skip", f"[robots-skip] {url}", url=url)
                    continue

                try:
                    resp = session.get(url, timeout=20)
                    resp.raise_for_status()
                except Exception as exc:
                    _emit_event(
                        event_callback,
                        "fetch_error",
                        f"[fetch-error] {url} -> {exc}",
                        url=url,
                        error=str(exc),
                    )
                    continue

                content_type = (resp.headers.get("content-type") or "").lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    _emit_event(
                        event_callback,
                        "skip_non_html",
                        f"[skip-non-html] {url} content-type={content_type or 'unknown'}",
                        url=url,
                        content_type=content_type,
                    )
                    continue

                doc, soup = _extract_document(url, resp.text)
                if doc is None:
                    _emit_event(
                        event_callback,
                        "skip_content",
                        f"[skip-content] {url} (insufficient extractable text)",
                        url=url,
                    )
                    continue

                inserted = _ingest_document(cur, embedder, doc)
                if raw_dir is not None:
                    _write_raw_snapshot(raw_dir, doc, resp.text)
                conn.commit()

                pages_ingested += 1
                chunk_count += inserted
                _emit_event(
                    event_callback,
                    "ingested",
                    f"[ingested] depth={depth} chunks={inserted} url={url}",
                    url=url,
                    depth=depth,
                    chunks=inserted,
                    pages_total=pages_ingested,
                    chunks_total=chunk_count,
                )

                if max_depth < 0 or depth < max_depth:
                    for link in _extract_links(soup, url, allowed_domains):
                        if link in visited or link in queued:
                            continue
                        if _is_author_based_url(link):
                            continue
                        queue.append((link, depth + 1))
                        queued.add(link)

                if delay_seconds > 0:
                    time.sleep(delay_seconds)

    interrupted = bool(stop_event is not None and stop_event.is_set())
    _emit_event(
        event_callback,
        "done",
        f"[done] pages={pages_ingested} chunks={chunk_count} interrupted={interrupted}",
        pages_total=pages_ingested,
        chunks_total=chunk_count,
        interrupted=interrupted,
    )
    return pages_ingested, chunk_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl scholarly domains and ingest into pgvector.")
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Seed URL to begin crawling. Pass multiple times for multiple seeds.",
    )
    parser.add_argument(
        "--allow-domain",
        action="append",
        default=[],
        help="Allowed domain suffix (for example: arxiv.org). Pass multiple times.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=int(os.getenv("SPIDER_MAX_PAGES", "0")),
        help="Maximum pages to ingest. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=int(os.getenv("SPIDER_MAX_DEPTH", "-1")),
        help="Maximum crawl depth. Use -1 for unlimited.",
    )
    parser.add_argument("--delay-seconds", type=float, default=float(os.getenv("SPIDER_DELAY_SECONDS", "1.0")))
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Disable robots.txt checks. Use with care.",
    )
    parser.add_argument(
        "--raw-dir",
        default=os.getenv("CRAWL_RAW_DIR", "").strip(),
        help="Optional folder to store raw HTML + metadata snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    seeds = [s for s in (args.seed or []) if str(s).strip()]
    if not seeds:
        seeds = [DEFAULT_SEED_URL]

    allowed_domains = [d.strip().lower() for d in args.allow_domain if d.strip()]
    if not allowed_domains:
        allowed_domains = list(DEFAULT_ALLOWED_DOMAINS)
    if not allowed_domains:
        allowed_domains = sorted(
            {
                (urlparse(s).hostname or "").lower()
                for s in seeds
                if _canonicalize_url(s) is not None
            }
        )

    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else None
    pages, chunks = crawl_and_ingest(
        seed_urls=seeds,
        allowed_domains=allowed_domains,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay_seconds,
        respect_robots=not args.ignore_robots,
        raw_dir=raw_dir,
    )
    print(f"Crawl complete. Pages ingested: {pages}. New chunks inserted: {chunks}.")


if __name__ == "__main__":
    main()
