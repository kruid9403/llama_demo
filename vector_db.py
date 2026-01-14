import numpy as np
import psycopg
import threading

from typing import List
from transformers import TextIteratorStreamer

from messages.system_prompt import get_system_prompt
from llama_demo import load_model, _compute_eot_ids

# =========================
# CONFIG
# =========================

DB_URL = "postgresql://dev_user:dev_password@localhost:5432/embedding_db"
EMBEDDING_DIM = 384
TOP_K = 8
MAX_NEW_TOKENS = 512

# =========================
# EMBEDDING (stub – replace later)
# =========================

def embed_query(text: str) -> List[float]:
    # Replace later with a real local embedding model
    return np.random.rand(EMBEDDING_DIM).tolist()

# =========================
# RETRIEVAL (pgvector)
# =========================

def similarity_search(query: str, limit: int = TOP_K):
    query_embedding = embed_query(query)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    content,
                    embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY distance
                LIMIT %s;
                """,
                (query_embedding, limit),
            )
            return cur.fetchall()

# =========================
# PROMPT CONSTRUCTION
# =========================

def build_prompt(system_prompt: str, question: str, retrieved_chunks: List[str]) -> str:
    context_block = "\n\n".join(
        f"[Context {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    return f"""{system_prompt}

You are answering using retrieved technical documentation.

Question:
{question}

Retrieved Context:
{context_block}

Answer clearly and concisely. Cite the context when relevant.
"""

# =========================
# GENERATION
# =========================

def generate_answer(model, tokenizer, prompt: str):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    eos_ids = _compute_eot_ids(tokenizer) or [tokenizer.eos_token_id]

    generate_kwargs = dict(
        inputs=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        do_sample=True,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs=generate_kwargs,
        daemon=True,
    )
    thread.start()

    for token in streamer:
        print(token, end="", flush=True)

    print("\n\n--- End of response ---\n")

# =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":
    system_prompt = get_system_prompt()

    model, tokenizer = load_model()
    print(f"Model loaded on device: {model.device}")

    while True:
        question = input(">>> ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        print("\n[retrieving relevant context...]\n")

        results = similarity_search(question)
        retrieved_chunks = [row[0] for row in results]

        prompt = build_prompt(
            system_prompt=system_prompt,
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        generate_answer(model, tokenizer, prompt)
