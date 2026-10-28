"""Replay a trace against an OpenAI-compatible server and record timings + cache reuse.

Each recorded call is sent as-is (same messages), in order, one at a time. Fixed tool outputs
mean this measures the serving side of the recorded run, not a fresh agent run.

Cache data comes from whatever the server returns on the final streamed chunk:
  - llama-server: `timings` {cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms}
  - OpenAI-style: usage.prompt_tokens_details.cached_tokens
Both are stored raw so the report can pick; nothing is inferred when a server gives neither.
"""
import json
import os
import time

import httpx

from agentcost.trace import read_trace


def target_max_tokens(turn):
    """Cap decode at the recorded completion length so decode work is comparable across runs."""
    if turn.get("usage"):
        return max(1, turn["usage"]["completion_tokens"])
    return max(1, len(turn["completion"] or "") // 4)  # ~4 chars/token; imported trajectories carry no usage


def replay_turn(client, endpoint, model, turn):
    body = {
        "model": model or turn.get("model") or "default",
        "messages": turn["messages"],
        "max_tokens": target_max_tokens(turn),
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": True,  # llama-server flag; other servers ignore it
    }
    t0 = time.perf_counter()
    ttft = None
    usage = timings = None
    n_chunks = 0
    with client.stream("POST", endpoint.rstrip("/") + "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            ch = ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                n_chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {
        "turn": turn["turn"],
        "max_tokens": body["max_tokens"],
        "max_tokens_source": "recorded_usage" if turn.get("usage") else "chars_div_4",
        "ttft_s": ttft,
        "total_s": total,
        "usage": usage,
        "timings": timings,
        "content_chunks": n_chunks,
    }


def flush_cache(client, endpoint):
    """Cold-start the server's prompt cache so a trace's first turn can't hit a previous run.

    Only llama-server (started with --slot-save-path) supports this. For vLLM/SGLang, restart the
    server between traces instead. Returns True if every slot was erased.
    """
    base = endpoint.rstrip("/")
    try:
        slots = client.get(base + "/slots").json()
        return all(client.post(f"{base}/slots/{s['id']}?action=erase").status_code == 200 for s in slots)
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        return False


def replay_trace(path, endpoint, out_dir, model=None, timeout=600):
    turns = read_trace(path)
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, os.path.basename(path))
    # No keep-alive: llama-server dropped the reused connection after a slow cold first turn
    # ("Server disconnected without sending a response"); one connection per request avoids it.
    with httpx.Client(timeout=timeout, limits=httpx.Limits(max_keepalive_connections=0)) as client, open(dest, "w") as f:
        if not flush_cache(client, endpoint):
            print("WARNING: could not clear server prompt cache; first-turn cache hits may come from a previous run")
        for t in turns:
            row = replay_turn(client, endpoint, model, t)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{os.path.basename(path)} turn {row['turn']}: ttft={row['ttft_s']} total={row['total_s']:.2f}s timings={row['timings']}")
    return dest
