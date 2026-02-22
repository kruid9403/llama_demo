import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "using",
    "we",
    "what",
    "when",
    "where",
    "which",
    "with",
}

METRIC_KEYWORDS = {
    "accuracy": "predictive accuracy",
    "precision": "precision/recall balance",
    "recall": "precision/recall balance",
    "f1": "overall F1 score",
    "latency": "inference latency",
    "throughput": "system throughput",
    "robust": "robustness under distribution shift",
    "robustness": "robustness under distribution shift",
    "generalization": "cross-domain generalization",
    "safety": "operational safety",
    "alignment": "behavioral alignment",
    "interpretability": "model interpretability",
    "energy": "compute efficiency",
}

CONTRIVED_PATTERNS = (
    r"\bcombine\b",
    r"\bcombining\b",
    r"\bintegrate\b",
    r"\bintegrating\b",
    r"\bhybrid\b",
    r"\bfusion\b",
    r"\bmerge\b",
    r"\bmerged\b",
    r"\bblending\b",
    r"\bblend\b",
    r"\bstitch\b",
    r"\bstitched\b",
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")]


def _key_terms(text: str, limit: int = 10) -> list[str]:
    counts = Counter(t for t in _tokenize(text) if t not in STOPWORDS and not t.isdigit())
    return [term for term, _ in counts.most_common(limit)]


def _metric_focus(prompt: str) -> str:
    prompt_l = (prompt or "").lower()
    for key, value in METRIC_KEYWORDS.items():
        if key in prompt_l:
            return value
    return "generalization and reliability"


def review_prompt_for_similarity(prompt: str) -> dict[str, Any]:
    cleaned = _normalize_space(prompt)
    quoted_phrases = [_normalize_space(p) for p in re.findall(r'"([^"]+)"', cleaned) if _normalize_space(p)]
    terms = _key_terms(cleaned, limit=12)

    query_parts: list[str] = []
    if quoted_phrases:
        query_parts.extend(quoted_phrases[:3])
    query_parts.extend(terms[:8])
    if not query_parts and cleaned:
        query_parts.append(cleaned)
    search_query = _normalize_space(" ".join(query_parts))

    notes: list[str] = []
    warnings: list[str] = []
    if quoted_phrases:
        notes.append("Prioritized quoted phrases from the prompt as intent anchors.")
    if terms:
        notes.append("Expanded query with high-signal topical terms from the prompt.")
    if len(terms) < 4:
        warnings.append("Prompt appears short; retrieval may be broad. Add domain/context constraints for precision.")
    if "author" in cleaned.lower():
        warnings.append("Author-oriented wording detected; query was normalized toward topical retrieval terms.")

    return {
        "cleaned_prompt": cleaned,
        "search_query": search_query,
        "key_terms": terms,
        "notes": notes,
        "warnings": warnings,
    }


def _domain_from_url(url: str | None) -> str:
    if not url:
        return ""
    return (urlparse(url).hostname or "").lower()


def summarize_references(results: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for idx, item in enumerate(results[:limit], start=1):
        content = _normalize_space(str(item.get("content", "")))
        excerpt = content[:320] + "..." if len(content) > 320 else content
        refs.append(
            {
                "id": idx,
                "distance": item.get("distance"),
                "weighted_distance": item.get("weighted_distance"),
                "citation_count": item.get("citation_count") or 0,
                "title": item.get("document") or "Untitled",
                "section": item.get("section") or "",
                "url": item.get("url") or "",
                "source_url": item.get("source_url") or "",
                "doi": item.get("doi") or "",
                "venue": item.get("venue") or "",
                "published_date": item.get("published_date") or "",
                "authors": item.get("authors") or "",
                "domain": _domain_from_url(item.get("source_url") or item.get("url")),
                "excerpt": excerpt,
            }
        )
    return refs


def build_result_context_blocks(results: list[dict[str, Any]], max_chars: int = 24000) -> str:
    if max_chars < 2000:
        max_chars = 2000
    blocks: list[str] = []
    used = 0
    for idx, item in enumerate(results, start=1):
        content = _normalize_space(str(item.get("content", "")))
        if not content:
            continue
        citation_count = int(item.get("citation_count") or 0)
        header_lines = [
            f"[Result {idx}]",
            f"Title: {item.get('document') or 'Untitled'}",
            f"URL: {item.get('url') or ''}",
            f"DOI: {item.get('doi') or ''}",
            f"Venue: {item.get('venue') or ''}",
            f"Citation count (in-corpus): {citation_count}",
            "Content:",
        ]
        header = "\n".join(header_lines) + "\n"
        available = max_chars - used - len(header) - 2
        if available <= 0:
            break
        if len(content) > available:
            content = content[: max(0, available - 3)] + "..."
        block = header + content
        blocks.append(block)
        used += len(block) + 2
        if used >= max_chars:
            break
    if not blocks:
        return "No retrieved context blocks were available."
    return "\n\n".join(blocks)


def build_llama_hypothesis_prompt(
    user_prompt: str,
    search_query: str,
    context_blocks: str,
    hypothesis_id: str,
) -> str:
    return (
        "You are a research scientist generating novel, testable hypotheses from retrieved literature context.\n"
        "Write as a careful scientist: explicit assumptions, mechanism-oriented reasoning, and falsifiable claims.\n"
        "Do not mention limitations about being an AI model.\n\n"
        "Task requirements:\n"
        "1. Produce one forward-looking hypothesis that is distinct and non-generic.\n"
        "2. The hypothesis must be a single original mechanism, not a stitched combination of named existing theories.\n"
        "3. Do not frame the core idea as 'combining/integrating/fusing' prior methods.\n"
        "4. Use prior work only as boundary conditions, failure modes, or evidence gaps.\n"
        "5. Provide why this hypothesis is plausible given the context.\n"
        "6. Provide a concrete research plan with experiments, baselines, and measurable outcomes.\n"
        "7. Cite supporting retrieved results by index like [Result 1], [Result 2].\n"
        "8. Include the hypothesis identifier exactly once: "
        f"{hypothesis_id}\n\n"
        f"Original user prompt:\n{_normalize_space(user_prompt)}\n\n"
        f"Derived similarity query:\n{_normalize_space(search_query)}\n\n"
        "Retrieved context (full result blocks):\n"
        f"{context_blocks}\n\n"
        "Output format:\n"
        "Hypothesis ID: <id>\n"
        "Forward-Looking Hypothesis: <1 paragraph>\n"
        "Scientific Rationale: <1 short paragraph>\n"
        "Research Program:\n"
        "- Experiment 1: ...\n"
        "- Experiment 2: ...\n"
        "- Experiment 3: ...\n"
        "Expected Impact: <1 short paragraph>\n"
    )


def is_contrived_hypothesis(text: str) -> bool:
    normalized = _normalize_space(text)
    if not normalized:
        return True

    # Validate the core hypothesis section when available.
    match = re.search(
        r"forward-looking hypothesis\s*:\s*(.+?)(?:\n[A-Z][A-Za-z ]{2,40}\s*:|\Z)",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        normalized = _normalize_space(match.group(1))

    lowered = normalized.lower()
    if not lowered:
        return True
    for pattern in CONTRIVED_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def _supporting_concepts(results: list[dict[str, Any]], limit: int = 8) -> list[str]:
    blob_parts: list[str] = []
    for item in results[:6]:
        blob_parts.append(str(item.get("document", "")))
        blob_parts.append(str(item.get("section", "")))
        blob_parts.append(str(item.get("content", ""))[:900])
    return _key_terms(" ".join(blob_parts), limit=limit)


def build_forward_hypothesis(
    prompt: str,
    reviewed_query: str,
    references: list[dict[str, Any]],
) -> str:
    if not references:
        return (
            "Forward-looking hypothesis: Current indexed evidence is insufficient for a grounded hypothesis. "
            "Collect additional domain-specific documents, then rerun retrieval with tighter topical constraints."
        )

    concepts = _supporting_concepts(references, limit=8)
    concept_a = concepts[0] if len(concepts) > 0 else "representation learning"
    concept_b = concepts[1] if len(concepts) > 1 else "structured retrieval"
    concept_c = concepts[2] if len(concepts) > 2 else "adaptive optimization"
    metric = _metric_focus(prompt)

    refs_hint = ", ".join(f"[{r['id']}]" for r in references[:3]) or "[1]"
    hypothesis = (
        "Forward-looking hypothesis: "
        f"A latent causal controller grounded in {concept_a}, constrained by {concept_b}, and adapted through {concept_c} "
        f"for the query scope '{reviewed_query}' will improve {metric} versus static baselines under cross-domain evaluation. "
        f"This direction is supported by the retrieved evidence set {refs_hint}. "
        "Recommended next experiments: (1) ablation on each component, (2) robustness tests on out-of-distribution "
        "samples, (3) compute/quality trade-off profiling for practical deployment."
    )
    return hypothesis
