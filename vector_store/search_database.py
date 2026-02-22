import argparse
import json


def _trim(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _run_query(
    query: str,
    top_k: int,
    recent_only: bool,
    urls: list[str] | None,
    as_json: bool,
    max_chars: int,
) -> None:
    from vector_store.retrieval import similarity_search_with_references

    results = similarity_search_with_references(
        question=query,
        top_k=top_k,
        recent_only=recent_only,
        document_urls=urls,
    )

    if as_json:
        print(json.dumps(results, ensure_ascii=True, indent=2))
        return

    if not results:
        print("No matches found.")
        return

    print(f"\nQuery: {query}")
    print(f"Matches: {len(results)}\n")
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] distance={item['distance']:.6f}")
        if item.get("weighted_distance") is not None:
            print(f"    weighted_distance: {float(item['weighted_distance']):.6f}")
        print(f"    citation_count: {int(item.get('citation_count') or 0)}")
        print(f"    title: {item.get('document') or '(untitled)'}")
        print(f"    section: {item.get('section') or '(none)'}")
        print(f"    url: {item.get('url') or '(none)'}")
        print(f"    source_url: {item.get('source_url') or '(none)'}")
        print(f"    doi: {item.get('doi') or '(none)'}")
        print(f"    venue: {item.get('venue') or '(none)'}")
        print(f"    published_date: {item.get('published_date') or '(none)'}")
        print(f"    authors: {item.get('authors') or '(none)'}")
        print(f"    excerpt: {_trim(item.get('content') or '', max_chars)}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run similarity search over pgvector chunks and show citation references."
    )
    parser.add_argument("--query", default="", help="Search query text.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of results to return.")
    parser.add_argument(
        "--recent-only",
        action="store_true",
        help="Restrict search to RECENT_DOCS_LIMIT newest documents.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Optional URL filter. Pass multiple times to include multiple documents.",
    )
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=320,
        help="Maximum excerpt length in text output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    urls = [u.strip() for u in (args.url or []) if u.strip()] or None

    if args.query.strip():
        _run_query(
            query=args.query.strip(),
            top_k=max(1, args.top_k),
            recent_only=args.recent_only,
            urls=urls,
            as_json=args.json,
            max_chars=max(80, args.max_chars),
        )
        return

    print("Interactive similarity search. Type 'exit' to quit.")
    while True:
        query = input("query> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        _run_query(
            query=query,
            top_k=max(1, args.top_k),
            recent_only=args.recent_only,
            urls=urls,
            as_json=args.json,
            max_chars=max(80, args.max_chars),
        )


if __name__ == "__main__":
    main()
