# agent-cost-lab

**Status:** M1 (traces in), M2 (replay and measure) and M3 (`report.html` with cost estimates) work. M4 has one strategy (truncate old tool outputs). Numbers below are pipeline validation on a CPU, not a benchmark.

## Problem

Agent workloads look nothing like chat: long contexts, many tool calls, and the same prefix resent on every call. Teams find out what an agent task costs when the bill arrives. Context choices (how much history to resend, what to truncate) are cost decisions that nobody measures as such.

Question this answers: *what does one agent task cost, how much of that does prefix caching save, and what would a different context strategy change?*

## What it measures

For each recorded LLM call in a task, replayed in order against an OpenAI-compatible server:

- prompt tokens, split into cached (`cache_n`) and freshly computed (`prompt_n`)
- time to first token (TTFT) and total latency
- prefill vs decode time, when the server reports it

`report.html` also gives an estimated cost per task: self-hosted (GPU $/hr x measured time) vs API (prices from a JSON file you edit), plus a short auto-written "why". `--strategy truncate` reruns the same trace with older tool outputs cut to 80 characters (the newest one stays whole), and the report compares the runs side by side.

## Run it (under 5 minutes, plus model download)

```bash
uv sync
# 1. a trace: the repo bundles data/tiny.jsonl, or convert SWE-agent trajectories
uv run agentcost import-sweagent path/to/trajs --out traces
# 2. a server: llama.cpp's llama-server (build from source: cmake -B build -DLLAMA_CURL=OFF && cmake --build build --target llama-server)
llama-server -m model.gguf -c 32768 --port 8080 --metrics --slot-save-path ./slots
# 3. replay
uv run agentcost replay data/tiny.jsonl --endpoint http://localhost:8080 --gpu-price 2.0
```

That writes `report.html`. For API cost, copy `prices.example.json` to `prices.json`, fill in $/million tokens from your provider (the file ships with nulls; the report says "not set" rather than guessing), and add `--prices prices.json --model <name>`. `agentcost report runs/*.jsonl` rebuilds the page without replaying.

Output goes to `runs/<strategy>/<trace>.jsonl`, one row per call. To record your own agent instead, run `agentcost proxy --upstream <server> --out traces/task1.jsonl` and point the agent at the proxy.

## Example result

`data/tiny.jsonl` (5 calls, from SWE-agent's HumanEvalFix demo) on Qwen2.5-0.5B Q4, llama-server, 4 CPU cores, cold cache:

| turn | cached tokens | computed tokens | TTFT |
|---|---|---|---|
| 0 | 0 | 1928 | 7.2 s |
| 1 | 1928 | 117 | 0.62 s |
| 2 | 2045 | 410 | 1.97 s |
| 3 | 2455 | 471 | 2.38 s |
| 4 | 2926 | 101 | 0.69 s |

Reading it: TTFT tracks the *computed* tokens, not the prompt length. Prefill ran at about 200 tokens/s, so turn 0 paid 7 s to read the whole prompt once and later turns paid only for what was new. Decode ran at about 32 tokens/s. On this setup prefill dominates time-to-first-token; the report states per trace whether prefill or decode dominated (here prefill: 63% of server time, 37% decode). Cost for this trace at a made-up $2/hr GPU is about a cent; that figure only shows the arithmetic, not a real price.

This is a 0.5B model on CPU. It validates the pipeline and the cache mechanics. It says nothing about GPU timings.

## Example: truncating old tool outputs

Same trace and setup, `--strategy full` vs `--strategy truncate` (one run each):

| | prompt tokens (sum) | cache hit | computed tokens (sum) | total time |
|---|---|---|---|---|
| full | 12381 | 76% | 3027 | 23.0 s |
| truncate | 11385 | 73% | 3074 | 21.8 s |

Truncation sent 8% fewer prompt tokens but computed slightly *more* of them. Rewriting an old message changes the middle of the prompt, so the server's prefix cache is only valid up to that message and everything after it is recomputed (turn 3: 2117 tokens cached vs 2455 for full history). The saving only wins when the text removed is bigger than the cache that gets thrown away. On a 5-call trace with ~1 KB observations it is a wash, and the 5% time difference is inside what I'd expect from run-to-run noise (not measured; one run each). Whether truncation pays off needs long traces with large tool outputs. Task success is not measured.

## Caveats

- **Replay is not a re-run.** Tool outputs are fixed from the recording, so this measures the serving side of a recorded task. `max_tokens` is capped at the recorded completion length so decode work is comparable. Imported trajectories have no usage data, so that cap is estimated as characters / 4 (`max_tokens_source` in each row says which).
- **Short traces.** The SWE-agent trajectories top out around 2k to 14k prompt tokens. Long-context behaviour needs traces from the proxy on your own agent runs.
- **Cache numbers differ per server.** llama-server returns `timings.cache_n` and `prompt_n` per request. Cumulative `/metrics` counters exist but cannot be attributed to a request. vLLM and SGLang expose other counters; those are not wired up or verified yet.
- **Cold start.** Replay erases llama-server's slots before each trace, which needs `--slot-save-path`. Otherwise it warns, and first-turn cache hits may come from an earlier run. For other servers, restart between traces.
- **Task success is not measured.** Anything about quality under a context strategy is out of scope.

See `predictions.md` for the predict-then-measure log and `IDEAS.md` for parked ideas.
