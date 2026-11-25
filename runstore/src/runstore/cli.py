"""``uv run runstore merge SRC DST``: fold one run store into another."""

from __future__ import annotations

import argparse

from runstore.store import merge_store


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="runstore", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    merge = sub.add_parser("merge", help="copy every run in SRC into DST (refuses id clashes)")
    merge.add_argument("src")
    merge.add_argument("dst")
    args = parser.parse_args(argv)
    if args.cmd == "merge":
        for run_id in merge_store(args.src, args.dst):
            print(f"merged {run_id}")


if __name__ == "__main__":
    main()
