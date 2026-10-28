"""Whole pipeline on the bundled trace, against a fake llama-server-style endpoint."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agentcost import replay
from agentcost.trace import read_trace

TRACE = Path(__file__).parent.parent / "data" / "tiny.jsonl"


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, json.dumps([{"id": 0}]).encode())

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if "/slots/" in self.path:
            return self._send(200, b"{}")
        chunks = [
            {"choices": [{"delta": {"content": "hi"}}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 2},
             "timings": {"cache_n": 60, "prompt_n": 40, "prompt_ms": 80.0, "predicted_n": 2, "predicted_ms": 20.0}},
        ]
        out = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        assert body["stream"] and body["max_tokens"] >= 1
        self._send(200, out.encode(), "text/event-stream")


def test_replay_tiny_trace(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = replay.replay_trace(str(TRACE), f"http://127.0.0.1:{srv.server_port}", str(tmp_path))
    finally:
        srv.shutdown()
    rows = read_trace(dest)
    assert len(rows) == len(read_trace(TRACE)) == 5
    assert all(r["ttft_s"] is not None and r["timings"]["cache_n"] == 60 for r in rows)
