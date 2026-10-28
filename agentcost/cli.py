import argparse

from agentcost import importers, proxy, replay, report, strategies


def main():
    ap = argparse.ArgumentParser(prog="agentcost")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("proxy", help="record an agent's LLM calls to a JSONL trace")
    p.add_argument("--upstream", required=True, help="server root, e.g. http://localhost:8080")
    p.add_argument("--out", required=True, help="trace file to append to (one task per file)")
    p.add_argument("--port", type=int, default=9000)

    p = sub.add_parser("import-sweagent", help="convert SWE-agent .traj files to traces")
    p.add_argument("src", help=".traj file or directory")
    p.add_argument("--out", default="traces")

    p = sub.add_parser("replay", help="replay traces against an OpenAI-compatible server, record timings")
    p.add_argument("traces", nargs="+", help="trace .jsonl files")
    p.add_argument("--endpoint", required=True, help="server root, e.g. http://localhost:8080")
    p.add_argument("--model", help="model name to send (default: recorded, else 'default')")
    p.add_argument("--out", default="runs")
    p.add_argument("--strategy", choices=sorted(strategies.STRATEGIES), default="full", help="context strategy applied before replay")
    p.add_argument("--gpu-price", type=float, help="USD per GPU hour, for the self-hosted cost estimate")
    p.add_argument("--prices", help="prices JSON (see prices.example.json) for the API cost estimate")
    p.add_argument("--report", default="report.html")

    p = sub.add_parser("report", help="rebuild report.html from existing runs/*.jsonl")
    p.add_argument("runs", nargs="+", help="replay output .jsonl files")
    p.add_argument("--gpu-price", type=float)
    p.add_argument("--prices")
    p.add_argument("--model", help="key into the prices file")
    p.add_argument("--out", default="report.html")

    a = ap.parse_args()
    if a.cmd == "proxy":
        proxy.serve(a.upstream, a.out, a.port)
    elif a.cmd == "import-sweagent":
        for path in importers.import_sweagent(a.src, a.out):
            print(path)
    elif a.cmd == "replay":
        runs = [replay.replay_trace(path, a.endpoint, a.out, a.model, strategy=a.strategy) for path in a.traces]
        print(report.build_report(runs, a.report, a.gpu_price, a.prices, a.model))
    elif a.cmd == "report":
        print(report.build_report(a.runs, a.out, a.gpu_price, a.prices, a.model))


if __name__ == "__main__":
    main()
