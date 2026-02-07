# nodes.py
from retrieval import retrieve_chunks
from arxiv_ingest import ingest_query
from url_loader import load_url_chunks
from state import ResearchState
from llama_demo import load_model, _compute_eot_ids
from transformers import TextIteratorStreamer
import threading


model, tokenizer = load_model()

STATUS_PREFIX = "[[STATUS]]"

def _emit_status(state: ResearchState, message: str) -> None:
    stream_queue = state.get("stream_queue")
    if stream_queue is None:
        print(f"[status] {message}")
        return
    stream_queue.put(f"\n{STATUS_PREFIX} {message}\n")

def retrieval_node(state: ResearchState) -> ResearchState:
    _emit_status(state, "Querying arXiv and updating the vector database...")
    try:
        ingest_query(state["question"])
    except Exception as exc:
        print(f"[arxiv] ingest failed: {exc}")
    _emit_status(state, "Retrieving context...")
    retrieved = retrieve_chunks(state["question"])
    url = state.get("url")
    if url:
        try:
            url_chunks = load_url_chunks(url)
        except Exception as exc:
            url_chunks = [
                {
                    "content": f"Failed to load URL {url}: {exc}",
                    "document": url,
                    "section": "URL load error",
                }
            ]
        retrieved = url_chunks + retrieved
    state["retrieved"] = retrieved
    return state

def rerank_node(state: ResearchState) -> ResearchState:
    """
    No-op reranking node.
    This is a placeholder so graph.py can import it cleanly.
    Later you can replace this with a real reranker.
    """
    return state

def generation_node(state: ResearchState) -> ResearchState:
    stream_queue = state.get("stream_queue")
    if stream_queue is None:
        print("Generating answer...")
    else:
        stream_queue.put(f"\n{STATUS_PREFIX} Generating answer...\n")
    history_text = ""
    if state.get("history"):
        history_text = "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in state["history"]
        )
    context = "\n\n".join(
        f"[{i+1}] {c['content']}\n(Source: {c['document']} – {c['section']})"
        for i, c in enumerate(state["retrieved"])
    )
    references = "\n".join(
        f"[{i+1}] {c['document']} — {c['section']}"
        for i, c in enumerate(state["retrieved"])
    )

    prompt = f"""
        You are a research assistant.

        Answer the question using ONLY the context below.
        Cite sources by number when relevant (e.g., [1]).
        If you include a references section, use the same numbers.

        Conversation so far:
        {history_text}

        Question:
        {state['question']}

        Context:
        {context}
        """
    if stream_queue is None:
        print(prompt)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    eos_ids = _compute_eot_ids(tokenizer) or [tokenizer.eos_token_id]

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            inputs=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            streamer=streamer,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        ),
        daemon=True,
    )
    thread.start()

    answer = ""
    for t in streamer:
        if stream_queue is None:
            print(t, end="", flush=True)
        else:
            stream_queue.put(t)
        answer += t

    if references:
        references_block = f"\n\nReferences:\n{references}"
        answer = f"{answer}{references_block}"
        if stream_queue is None:
            print(references_block)
        else:
            stream_queue.put(references_block)

    if stream_queue is None:
        print("\n\n--- End of response ---\n")
    else:
        stream_queue.put(None)

    state["answer"] = answer
    return state
