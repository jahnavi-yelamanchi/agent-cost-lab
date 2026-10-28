"""Context strategies: rewrite each recorded call's messages before replay.

truncate: keep the newest KEEP_LAST tool outputs whole; cut older ones to HEAD_CHARS + a marker.
A "tool output" is any message after the system prompt and the first user message (the task) whose
role is user or tool; in SWE-agent traces the observations come back as user messages.

Note this edits text in the middle of the prompt, so the server's prefix cache is invalidated from the
first edited message onward. Whether the saved tokens beat the lost cache hits is the thing to measure.
Task success under truncation is NOT measured: the recorded completions are fixed.
"""
KEEP_LAST = 1
HEAD_CHARS = 80


def truncate_messages(messages):
    idx = [i for i, m in enumerate(messages) if i >= 2 and m["role"] in ("user", "tool")]
    out = [dict(m) for m in messages]
    for i in idx[:-KEEP_LAST]:
        c = out[i]["content"] or ""
        if len(c) > HEAD_CHARS:
            out[i]["content"] = c[:HEAD_CHARS] + f"\n[... {len(c) - HEAD_CHARS} chars truncated]"
    return out


STRATEGIES = {"full": lambda messages: messages, "truncate": truncate_messages}


def apply_strategy(name, turns):
    return [{**t, "messages": STRATEGIES[name](t["messages"])} for t in turns]
