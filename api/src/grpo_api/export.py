"""Snapshot the API to static JSON so the dashboard can be hosted with no server.

``uv run export-static [OUT]`` (default ``web/public/data``) requests every API endpoint
in-process and writes each response body unchanged to ``OUT/<path>.json``; for example
``/runs/<id>/rollouts/10`` becomes ``OUT/runs/<id>/rollouts/10.json``. The web app's data
layer maps API paths to those files, so the hosted dashboard serves exactly what the API serves.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from grpo_api.main import create_app


def export_static(store_dir: str | Path, out_dir: str | Path) -> int:
    """Write every endpoint's response under ``out_dir``. Returns the number of files."""
    client = TestClient(create_app(str(store_dir)))
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)  # stale runs must not survive a re-export

    def save(path: str) -> bytes:
        resp = client.get(path)
        resp.raise_for_status()
        target = out / f"{path.lstrip('/')}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)
        return resp.content

    count = 0
    for run in client.get("/runs").json():
        run_id = run["id"]
        for path in (f"/runs/{run_id}", f"/runs/{run_id}/metrics"):
            save(path)
            count += 1
        save(f"/runs/{run_id}/rollouts")
        count += 1
        for step in client.get(f"/runs/{run_id}/rollouts").json():
            save(f"/runs/{run_id}/rollouts/{step}")
            count += 1
    save("/runs")  # last, so a partial export has no index pointing at missing files
    return count + 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="export-static", description=__doc__)
    parser.add_argument("out", nargs="?", default="web/public/data")
    parser.add_argument("--store", default=os.environ.get("GRPO_STORE", "runs"))
    args = parser.parse_args(argv)
    n = export_static(args.store, args.out)
    print(f"wrote {n} files to {args.out}")


if __name__ == "__main__":
    main()
