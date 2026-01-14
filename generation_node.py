from llama_demo import load_model, _compute_eot_ids
from transformers import TextIteratorStreamer
import threading

def generation_node(state: ResearchState) -> ResearchState:
    model, tokenizer = load_model()

    context = "\n\n".join(
        f"[{i+1}] {c['content']}\n(Source: {c['document']} – {c['section']})"
        for i, c in enumerate(state["retrieved"])
    )

    prompt = f"""
You are a research assistant.

Answer the question using ONLY the context below.
Cite sources by number when relevant.

Question:
{state['question']}

Context:
{context}
"""
    print(prompt)

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    eos_ids = _compute_eot_ids(tokenizer) or [tokenizer.eos_token_id]

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            inputs=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        ),
        daemon=True,
    )
    thread.start()

    answer = ""
    for t in streamer:
        print(t, end="", flush=True)
        answer += t

    print("\n\n--- End of response ---\n")
    state["answer"] = answer
    return state
