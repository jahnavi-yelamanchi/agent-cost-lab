"""Trace format shared by the proxy, the importer and (later) replay.

A trace is one JSONL file = one agent task. Each line is one LLM call:
  {"turn": 0, "model": "...", "messages": [{"role", "content"}, ...],
   "completion": "...", "usage": {"prompt_tokens": N, "completion_tokens": N} | null}
"""
import json


def read_trace(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_trace(path, turns):
    with open(path, "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
