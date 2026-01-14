import io
import re
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

MAX_TEXT_CHARS = 12000
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_CHUNKS = 6


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _extract_html_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return _clean_text(text)


def _extract_pdf_text(content: bytes) -> str:
    return _clean_text(extract_text(io.BytesIO(content)))


def load_url_chunks(url: str) -> List[Dict[str, str]]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    content = response.content

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        text = _extract_pdf_text(content)
    else:
        text = _extract_html_text(content)

    if not text:
        return []

    text = text[:MAX_TEXT_CHARS]
    chunks = _chunk_text(text)
    return [
        {
            "content": chunk,
            "document": url,
            "section": f"URL chunk {idx + 1}",
        }
        for idx, chunk in enumerate(chunks)
    ]
