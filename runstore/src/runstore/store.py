"""Run store: the contract between the trainer (writer) and the API (reader).

On-disk layout under ``root``::

    store.sqlite                            runs + metrics (SQLite, WAL mode)
    rollouts/<run_id>/step_000010.parquet   logged rollout groups for one step

The trainer and API never talk to each other directly; both depend only on this module.
That lets the dashboard be built and deployed against recorded runs with no GPU attached.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1

ROLLOUT_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("step", pa.int64()),
        ("prompt_id", pa.string()),
        ("group_idx", pa.int64()),
        ("sample_idx", pa.int64()),
        ("prompt", pa.string()),
        ("completion", pa.string()),
        ("reward_total", pa.float64()),
        ("reward_components", pa.string()),  # JSON object: component name -> value
        ("gold_reward", pa.float64()),  # nullable; differs from reward when the verifier is a proxy
        ("advantage", pa.float64()),
        ("length", pa.int64()),
        ("truncated", pa.bool_()),
    ]
)
ROLLOUT_COLUMNS = tuple(ROLLOUT_SCHEMA.names)

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    git_sha TEXT,
    seed INTEGER,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(id),
    step INTEGER NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (run_id, key, step)
);
"""


@dataclass(frozen=True)
class RunInfo:
    id: str
    name: str
    config: dict[str, Any]
    git_sha: str | None
    seed: int | None
    status: str
    started_at: float
    finished_at: float | None


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "store.sqlite"
        with self._connect() as conn:
            conn.executescript(_DDL)
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation: safe across threads (FastAPI) and processes.
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writes (trainer) -------------------------------------------------------------------

    def create_run(
        self,
        name: str,
        config: Mapping[str, Any],
        seed: int | None = None,
        git_sha: str | None = None,
    ) -> str:
        run_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (id, name, config_json, git_sha, seed, status, started_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (run_id, name, json.dumps(config, default=str), git_sha, seed, time.time()),
            )
        return run_id

    def finish_run(self, run_id: str, status: str = "finished") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
                (status, time.time(), run_id),
            )

    def log_metrics(self, run_id: str, step: int, metrics: Mapping[str, float]) -> None:
        # Non-finite values are dropped: SQLite turns NaN into NULL, which would violate NOT NULL.
        rows = [(run_id, step, k, float(v)) for k, v in metrics.items() if math.isfinite(float(v))]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO metrics (run_id, step, key, value) VALUES (?, ?, ?, ?)",
                rows,
            )

    def log_rollouts(self, run_id: str, step: int, rows: Sequence[Mapping[str, Any]]) -> Path:
        records = [{**row, "run_id": run_id, "step": step} for row in rows]
        missing = (
            set(ROLLOUT_COLUMNS) - set().union(*(r.keys() for r in records)) if records else set()
        )
        if missing:
            raise ValueError(f"rollout rows missing columns: {sorted(missing)}")
        table = pa.Table.from_pylist(records, schema=ROLLOUT_SCHEMA)
        out_dir = self.root / "rollouts" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step_{step:06d}.parquet"
        pq.write_table(table, path, compression="zstd")
        return path

    # ---- reads (API) ------------------------------------------------------------------------

    def list_runs(self) -> list[RunInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, config_json, git_sha, seed, status, started_at, finished_at "
                "FROM runs ORDER BY started_at DESC"
            ).fetchall()
        return [_run_from_row(r) for r in rows]

    def get_run(self, run_id: str) -> RunInfo | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, config_json, git_sha, seed, status, started_at, finished_at "
                "FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return _run_from_row(row) if row else None

    def metric_keys(self, run_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT key FROM metrics WHERE run_id = ? ORDER BY key", (run_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def get_metrics(
        self, run_id: str, keys: Sequence[str] | None = None
    ) -> dict[str, list[tuple[int, float]]]:
        query = "SELECT key, step, value FROM metrics WHERE run_id = ?"
        params: list[Any] = [run_id]
        if keys:
            query += f" AND key IN ({','.join('?' * len(keys))})"
            params.extend(keys)
        query += " ORDER BY key, step"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out: dict[str, list[tuple[int, float]]] = {}
        for key, step, value in rows:
            out.setdefault(key, []).append((step, value))
        return out

    def rollout_steps(self, run_id: str) -> list[int]:
        run_dir = self.root / "rollouts" / run_id
        if not run_dir.is_dir():
            return []
        return sorted(int(p.stem.removeprefix("step_")) for p in run_dir.glob("step_*.parquet"))

    def read_rollouts(self, run_id: str, step: int) -> list[dict[str, Any]]:
        path = self.root / "rollouts" / run_id / f"step_{step:06d}.parquet"
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()


def _run_from_row(row: tuple[Any, ...]) -> RunInfo:
    run_id, name, config_json, git_sha, seed, status, started_at, finished_at = row
    return RunInfo(
        id=run_id,
        name=name,
        config=json.loads(config_json),
        git_sha=git_sha,
        seed=seed,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
    )
