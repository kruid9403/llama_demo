import json
import os
import queue
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

from graph import app


WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        if self.path not in {"/api/ask", "/api/stream"}:
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        question = (data.get("question") or "").strip()
        history = data.get("history") or []
        url = (data.get("url") or "").strip()
        if not question:
            self.send_error(400, "Missing question")
            return

        if self.path == "/api/ask":
            result = app.invoke(
                {
                    "question": question,
                    "retrieved": [],
                    "answer": "",
                "history": history,
                "url": url or None,
                }
            )
            answer = result.get("answer", "")

            payload = json.dumps({"answer": answer}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        token_queue = queue.Queue()

        def run_graph():
            try:
                app.invoke(
                    {
                        "question": question,
                        "retrieved": [],
                        "answer": "",
                        "history": history,
                        "url": url or None,
                        "stream_queue": token_queue,
                    }
                )
            finally:
                token_queue.put(None)

        threading.Thread(target=run_graph, daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

        try:
            while True:
                token = token_queue.get()
                if token is None:
                    break
                chunk = token.encode("utf-8")
                self.wfile.write(chunk)
                self.wfile.flush()
        except BrokenPipeError:
            return


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = HTTPServer((host, port), Handler)
    print(f"Web UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
