"""Import SWE-agent .traj files into the trace format.

A .traj stores the final message history. An agent makes one LLM call per assistant
message, and the prompt for that call is everything before it, so
turn i = (messages=history[:i], completion=history[i]).
"""
import json
import os

from agentcost.trace import write_trace


def sweagent_to_turns(path):
    with open(path) as f:
        history = json.load(f)["history"]
    # Keep role+content only. Function-calling variants also carry tool_calls, which we drop,
    # so their prompt token counts are slightly understated.
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    return [
        {"turn": n, "model": None, "messages": msgs[:i], "completion": msgs[i]["content"], "usage": None}
        for n, i in enumerate(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    ]


def import_sweagent(src, out_dir):
    """src: a .traj file or a directory searched recursively. Returns written paths."""
    if os.path.isdir(src):
        files = sorted(os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs if f.endswith(".traj"))
    else:
        files = [src]
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for k, p in enumerate(files):
        turns = sweagent_to_turns(p)
        if not turns:
            continue
        name = f"{k:03d}__" + os.path.basename(os.path.dirname(os.path.abspath(p)))[:40] + "__" + os.path.splitext(os.path.basename(p))[0]
        dest = os.path.join(out_dir, name.replace(" ", "_") + ".jsonl")
        write_trace(dest, turns)
        written.append(dest)
    return written
