"""Build a single static report.html from replay output (runs/*.jsonl).

Cost model:
  self-hosted = gpu $/hr / 3600 * sum(total_s). Assumes the GPU serves only this task, one call at a time.
  API         = (uncached prompt * input + cached prompt * cached_input + completion * output) / 1e6,
                with prices from the user's prices file (never hardcoded here).
"""
import html
import json
import os

from agentcost.trace import read_trace


def turn_stats(row):
    """Normalise one replay row. Cache fields are None when the server didn't report them."""
    t, u = row.get("timings") or {}, row.get("usage") or {}
    cached = t.get("cache_n")
    if cached is None:
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if "prompt_n" in t and cached is not None:
        computed = t["prompt_n"]
        prompt = cached + computed
    else:
        prompt = u.get("prompt_tokens")
        computed = None if prompt is None or cached is None else prompt - cached
    out = u.get("completion_tokens", t.get("predicted_n"))
    if "prompt_ms" in t and "predicted_ms" in t:
        prefill_s, decode_s, split = t["prompt_ms"] / 1000, t["predicted_ms"] / 1000, "server timings"
    else:  # proxy: TTFT is mostly prefill
        prefill_s = row["ttft_s"] or 0
        decode_s, split = row["total_s"] - prefill_s, "TTFT proxy"
    return {"turn": row["turn"], "prompt": prompt, "cached": cached, "computed": computed, "out": out,
            "hit": None if not prompt or cached is None else cached / prompt,
            "ttft": row["ttft_s"], "total": row["total_s"], "prefill": prefill_s, "decode": decode_s, "split": split}


def api_cost(stats, price):
    need = ("input_per_mtok", "cached_input_per_mtok", "output_per_mtok")
    if not price or any(price.get(k) is None for k in need):
        return None
    if any(s["computed"] is None or s["out"] is None for s in stats):
        return None
    c = sum(s["computed"] * price["input_per_mtok"] + s["cached"] * price["cached_input_per_mtok"]
            + s["out"] * price["output_per_mtok"] for s in stats)
    return c / 1e6


def why(stats):
    total = sum(s["total"] for s in stats)
    worst = max(stats, key=lambda s: s["total"])
    pre, dec = sum(s["prefill"] for s in stats), sum(s["decode"] for s in stats)
    dom = "prefill" if pre >= dec else "decode"
    lines = [f"Turn {worst['turn']} took the longest: {worst['total']:.1f}s, {worst['total'] / total:.0%} of task time"
             + (f", computing {worst['computed']} uncached prompt tokens." if worst["computed"] is not None else ".")]
    lines.append(f"{dom.capitalize()} dominated time: prefill {pre:.1f}s vs decode {dec:.1f}s "
                 f"({pre / (pre + dec):.0%} / {dec / (pre + dec):.0%}; source: {stats[0]['split']}).")
    hits = [s for s in stats if s["hit"] is not None]
    if hits:
        cached, prompt = sum(s["cached"] for s in hits), sum(s["prompt"] for s in hits)
        lines.append(f"{cached / prompt:.0%} of prompt tokens were served from cache across the task "
                     "(including the cold first turn).")
    else:
        lines.append("The server reported no cache data, so cache hit rate is unknown.")
    return lines


def fmt(v, spec="", dash="n/a"):
    return dash if v is None else format(v, spec)


def trace_section(name, stats, gpu_price, price):
    e = html.escape
    rows = "".join(
        f"<tr><td>{s['turn']}</td><td>{fmt(s['prompt'])}</td><td>{fmt(s['cached'])}</td><td>{fmt(s['hit'], '.0%')}</td>"
        f"<td>{fmt(s['out'])}</td><td>{fmt(s['ttft'], '.2f')}</td><td>{s['total']:.2f}</td>"
        f"<td>{s['prefill']:.2f}</td><td>{s['decode']:.2f}</td></tr>" for s in stats)
    secs = sum(s["total"] for s in stats)
    own = "not set (pass --gpu-price)" if gpu_price is None else f"${gpu_price / 3600 * secs:.4f}"
    api = api_cost(stats, price)
    api_s = "not set (fill in the prices file for this model)" if api is None else f"${api:.4f}"
    return (f"<h2>{e(name)}</h2><table><tr><th>turn</th><th>prompt tok</th><th>cached tok</th><th>hit rate</th>"
            f"<th>completion tok</th><th>TTFT s</th><th>total s</th><th>prefill s</th><th>decode s</th></tr>{rows}</table>"
            f"<p>Total time {secs:.1f}s. Estimated cost: self-hosted {own}; API {api_s}.</p>"
            f"<h3>Why</h3><ul>{''.join(f'<li>{e(l)}</li>' for l in why(stats))}</ul>")


CAVEATS = """<h2>Read this first</h2><ul>
<li>Replay sends recorded prompts with fixed tool outputs. It measures the serving side of a recorded task, not a fresh agent run.</li>
<li>Decode length is capped at the recorded completion length (estimated as characters / 4 when the recording has no usage data).</li>
<li>Self-hosted cost assumes the GPU serves only this task, one call at a time. Dev runs on CPU are pipeline validation, not GPU predictions.</li>
<li>Task success under a context strategy is not measured.</li></ul>"""

STYLE = ("body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
         "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.25rem .6rem;text-align:right}th{background:#f3f3f3}")


def run_label(path):
    """runs/<strategy>/<trace>.jsonl -> '<strategy>: <trace>.jsonl'."""
    return f"{os.path.basename(os.path.dirname(path))}: {os.path.basename(path)}"


def summary_table(labelled, gpu_price, price):
    """One row per run so strategies can be compared side by side."""
    rows = ""
    for label, stats in labelled:
        prompt, cached = sum(s["prompt"] or 0 for s in stats), sum(s["cached"] or 0 for s in stats)
        known = all(s["hit"] is not None for s in stats)
        secs = sum(s["total"] for s in stats)
        own = "n/a" if gpu_price is None else f"${gpu_price / 3600 * secs:.4f}"
        api = api_cost(stats, price)
        rows += (f"<tr><td style='text-align:left'>{html.escape(label)}</td><td>{prompt}</td>"
                 f"<td>{fmt(cached / prompt if known and prompt else None, '.0%')}</td>"
                 f"<td>{sum(s['computed'] or 0 for s in stats)}</td><td>{secs:.1f}</td><td>{own}</td>"
                 f"<td>{fmt(api, '$.4f')}</td></tr>")
    return ("<h2>Comparison</h2><table><tr><th>run</th><th>prompt tok (sum)</th><th>cache hit</th>"
            "<th>computed tok (sum)</th><th>total s</th><th>self-hosted</th><th>API</th></tr>" + rows + "</table>"
            "<p>Task success under a strategy is not measured: recorded completions are fixed, so this compares cost and latency only.</p>")


def build_report(run_files, out_path, gpu_price=None, prices_path=None, model=None):
    prices = {}
    if prices_path:
        with open(prices_path) as f:
            prices = json.load(f).get("models", {})
    price = prices.get(model) if model else None
    labelled = [(run_label(p), [turn_stats(r) for r in read_trace(p)]) for p in run_files]
    body = summary_table(labelled, gpu_price, price) + "".join(
        trace_section(label, stats, gpu_price, price) for label, stats in labelled)
    doc = (f"<!doctype html><meta charset=utf-8><title>agentcost report</title><style>{STYLE}</style>"
           f"<h1>agentcost report</h1>{CAVEATS}{body}")
    with open(out_path, "w") as f:
        f.write(doc)
    return out_path
