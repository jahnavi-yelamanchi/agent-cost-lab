"""Logging proxy: sits in front of any OpenAI-compatible server (upstream = server root, e.g. http://host:8080;
the request path, /v1/... included, is forwarded unchanged) and records every
/chat/completions call to one JSONL trace (one proxy run = one trace = one agent task).

Streaming is relayed as-is while we accumulate the text. We add stream_options.include_usage
to streaming requests that lack it, so the server reports token counts.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

# Request fields worth keeping so the trace can be replayed faithfully.
KEEP = ("model", "messages", "tools", "tool_choice", "max_tokens", "max_completion_tokens", "temperature")


def serve(upstream, out_path, port):
    upstream = upstream.rstrip("/")
    lock = threading.Lock()
    state = {"turn": 0}
    client = httpx.Client(timeout=None)

    def record(req, completion, tool_calls, usage, t0):
        with lock:
            row = {"turn": state["turn"], **{k: req[k] for k in KEEP if k in req},
                   "completion": completion, "usage": usage, "wall_s": round(time.time() - t0, 3)}
            if tool_calls:
                row["tool_calls"] = tool_calls
            state["turn"] += 1
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: response ends when we close the connection, so streams need no chunk framing.
        def log_message(self, *a):
            pass

        def _forward(self, method):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            req = None
            if method == "POST" and self.path.rstrip("/").endswith("/chat/completions"):
                req = json.loads(body)
                if req.get("stream"):
                    req.setdefault("stream_options", {}).setdefault("include_usage", True)
                    body = json.dumps(req).encode()
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() in ("authorization", "content-type", "accept")}
            t0 = time.time()
            with client.stream(method, upstream + self.path,
                               content=body or None, headers=headers) as r:
                self.send_response(r.status_code)
                self.send_header("Content-Type", r.headers.get("content-type", "application/json"))
                self.end_headers()
                raw = []
                for chunk in r.iter_bytes():
                    raw.append(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
            if req is not None and r.status_code == 200:
                record(req, *parse_response(b"".join(raw), bool(req.get("stream"))), t0)

        def do_POST(self):
            self._forward("POST")

        def do_GET(self):
            self._forward("GET")

    print(f"proxy :{port} -> {upstream}   recording to {out_path}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def parse_response(raw, stream):
    """-> (completion_text, tool_calls, usage)"""
    if not stream:
        d = json.loads(raw)
        msg = d["choices"][0]["message"]
        return msg.get("content") or "", msg.get("tool_calls"), d.get("usage")
    text, usage, calls = [], None, {}
    for line in raw.decode().splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage
        for c in d.get("choices") or []:
            delta = c.get("delta", {})
            text.append(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                cur = calls.setdefault(tc["index"], {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                cur["id"] = tc.get("id") or cur["id"]
                fn = tc.get("function", {})
                cur["function"]["name"] += fn.get("name") or ""
                cur["function"]["arguments"] += fn.get("arguments") or ""
    return "".join(text), [calls[k] for k in sorted(calls)] or None, usage
