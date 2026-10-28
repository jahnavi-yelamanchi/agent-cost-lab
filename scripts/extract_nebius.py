"""Extract one long trajectory from the nebius/SWE-agent-trajectories dataset (CC-BY-4.0) as a trace.

  uv run --with pyarrow python scripts/extract_nebius.py SHARD.parquet OUT.jsonl [--min-chars N] [--max-chars N]

SHARD.parquet: e.g. https://huggingface.co/datasets/nebius/SWE-agent-trajectories/resolve/main/data/train-00000-of-00012.parquet
Picks the trajectory with the most text within [min, max] chars and at most --max-messages messages
(first row group only). Capping messages skips agents looping on tiny commands and favours big tool outputs.
These are someone else's agent runs, not ours: use them to get long contexts, not to claim anything about an agent.
Same conversion rule as the SWE-agent importer: call i = (history[:i], history[i]) per assistant message.
"""
import argparse
import json

import pyarrow.parquet as pq

ROLE = {"ai": "assistant", "user": "user", "system": "system"}


def to_messages(trajectory):
    # The system message has empty `text`; its content is in `system_prompt`.
    return [{"role": ROLE[m["role"]], "content": (m["system_prompt"] if m["role"] == "system" else m["text"]) or ""}
            for m in trajectory]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("out")
    ap.add_argument("--min-chars", type=int, default=90_000)
    ap.add_argument("--max-chars", type=int, default=115_000)
    ap.add_argument("--max-messages", type=int, default=120)
    a = ap.parse_args()

    rows = pq.ParquetFile(a.shard).read_row_group(0, columns=["instance_id", "model_name", "trajectory"]).to_pylist()
    best = None
    for r in rows:
        msgs = to_messages(r["trajectory"])
        n = sum(len(m["content"]) for m in msgs)
        if a.min_chars <= n <= a.max_chars and len(msgs) <= a.max_messages and (best is None or n > best[2]):
            best = (r, msgs, n)
    if best is None:
        raise SystemExit("no trajectory in the requested size range")
    r, msgs, n = best
    with open(a.out, "w") as f:
        for k, i in enumerate(i for i, m in enumerate(msgs) if m["role"] == "assistant"):
            f.write(json.dumps({"turn": k, "model": None, "messages": msgs[:i], "completion": msgs[i]["content"],
                                "usage": None}) + "\n")
    print(f"{r['instance_id']} ({r['model_name']}): {len(msgs)} messages, {n} chars -> {a.out}")


if __name__ == "__main__":
    main()
