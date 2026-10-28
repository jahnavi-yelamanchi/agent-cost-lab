# Parked ideas

Not built. Add here instead of expanding the MVP.

- Summarize-history strategy (M4 does truncate-old-tool-outputs only, or the reverse; the other goes here).
- Measure task success under a context strategy (needs a verifier; out of MVP scope).
- vLLM / SGLang `/metrics` scraping for cache hit rate on a rented GPU.
- Concurrency: replay N traces in parallel to see cache eviction under load.
- Import more trajectory formats (OpenHands, Claude Code transcripts).
- Repeat each replay N times and report the spread, so small latency differences can be told from noise.
- Cache-friendly truncation: truncate in blocks, or truncate at write time, so edits don't move the cached prefix every turn.
