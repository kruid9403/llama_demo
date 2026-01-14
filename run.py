# run.py
from graph import app

import os
os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.7,max_split_size_mb:128"


if __name__ == "__main__":
    history = []
    while True:
        q = input(">>> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": q})
        result = app.invoke({
            "question": q,
            "retrieved": [],
            "answer": "",
            "history": history,
        })
        history.append({"role": "assistant", "content": result.get("answer", "")})
