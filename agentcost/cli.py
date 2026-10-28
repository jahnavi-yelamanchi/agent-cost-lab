import argparse

from agentcost import importers, proxy


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

    a = ap.parse_args()
    if a.cmd == "proxy":
        proxy.serve(a.upstream, a.out, a.port)
    elif a.cmd == "import-sweagent":
        for path in importers.import_sweagent(a.src, a.out):
            print(path)


if __name__ == "__main__":
    main()
