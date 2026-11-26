"""Run store: the contract between the trainer (writer) and the API (reader).

On-disk layout under ``root``::

    store.sqlite                                runs + metrics (SQLite, WAL mode)
    rollouts/<run_id>/step_000010.parquet       logged rollout groups for one step
    token_traces/<run_id>/step_000010.parquet   per-token values for a few of those groups

The trainer and API never talk to each other directly; both depend only on this module.
That lets the dashboard be built and deployed against recorded runs with no GPU attached.
"""

from __future__ import annotations

import json
import math
import shutil
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

# One row per traced sample, keyed like its rollout row; list columns hold one value per
# completion token (up to and including EOS). ref_logprob and kl are null when beta == 0.
TRACE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("step", pa.int64()),
        ("prompt_id", pa.string()),
        ("group_idx", pa.int64()),
        ("sample_idx", pa.int64()),
        ("token_ids", pa.list_(pa.int32())),
        ("tokens", pa.list_(pa.string())),  # each token decoded on its own
        ("logprob", pa.list_(pa.float32())),  # under the sampling policy
        ("ref_logprob", pa.list_(pa.float32())),
        ("kl", pa.list_(pa.float32())),  # k3 estimator against the reference
        ("entropy", pa.list_(pa.float32())),
        ("ratio", pa.list_(pa.float32())),  # pi / pi_old on the last pass; 1 when mu == 1
    ]
)
TRACE_COLUMNS = tuple(TRACE_SCHEMA.names)
_TRACE_LISTS = ("token_ids", "tokens", "logprob", "ref_logprob", "kl", "entropy", "ratio")
_NULLABLE_TRACE_LISTS = ("ref_logprob", "kl")

# Per-run directories of Parquet files, one file per logged step.
_RUN_FILE_TREES = ("rollouts", "token_traces")

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

    def log_traces(self, run_id: str, step: int, rows: Sequence[Mapping[str, Any]]) -> Path:
        records = [{**row, "run_id": run_id, "step": step} for row in rows]
        for r in records:
            missing = set(TRACE_COLUMNS) - set(r)
            if missing:
                raise ValueError(f"trace rows missing columns: {sorted(missing)}")
            n = len(r["token_ids"])
            bad = [
                k
                for k in _TRACE_LISTS
                if (r[k] is None and k not in _NULLABLE_TRACE_LISTS)
                or (r[k] is not None and len(r[k]) != n)
            ]
            if bad:
                raise ValueError(f"trace lists {bad} are null or don't match {n} token_ids")
        table = pa.Table.from_pylist(records, schema=TRACE_SCHEMA)
        out_dir = self.root / "token_traces" / run_id
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

    def last_metrics(self, run_id: str) -> dict[str, tuple[int, float]]:
        """The latest (step, value) of every metric key: a run's "final" numbers."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, step, value FROM metrics m WHERE run_id = ? AND step = "
                "(SELECT MAX(step) FROM metrics WHERE run_id = m.run_id AND key = m.key) "
                "ORDER BY key",
                (run_id,),
            ).fetchall()
        return {key: (step, value) for key, step, value in rows}

    def rollout_steps(self, run_id: str) -> list[int]:
        return _steps_in(self.root / "rollouts" / run_id)

    def read_rollouts(self, run_id: str, step: int) -> list[dict[str, Any]]:
        path = self.root / "rollouts" / run_id / f"step_{step:06d}.parquet"
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()

    def trace_steps(self, run_id: str) -> list[int]:
        return _steps_in(self.root / "token_traces" / run_id)

    def trace_groups(self, run_id: str, step: int) -> list[int]:
        path = self._trace_path(run_id, step)
        if not path.exists():
            return []
        return sorted(set(pq.read_table(path, columns=["group_idx"])["group_idx"].to_pylist()))

    def read_traces(
        self, run_id: str, step: int, group_idx: int | None = None
    ) -> list[dict[str, Any]]:
        path = self._trace_path(run_id, step)
        if not path.exists():
            return []
        filters = [("group_idx", "=", group_idx)] if group_idx is not None else None
        return pq.read_table(path, filters=filters).to_pylist()

    def _trace_path(self, run_id: str, step: int) -> Path:
        return self.root / "token_traces" / run_id / f"step_{step:06d}.parquet"


def _steps_in(run_dir: Path) -> list[int]:
    if not run_dir.is_dir():
        return []
    return sorted(int(p.stem.removeprefix("step_")) for p in run_dir.glob("step_*.parquet"))


def merge_store(src: str | Path, dst: str | Path) -> list[str]:
    """Copy every run (row, metrics, rollouts, token traces) from ``src`` into ``dst``.

    For stores written on different machines, e.g. runs trained on separate GPU pods. Refuses,
    before writing anything, if a run id already exists in ``dst``. Returns the merged run ids.
    """
    src_root, dst_store = Path(src), RunStore(dst)
    src_db = src_root / "store.sqlite"
    if not src_db.exists():
        raise FileNotFoundError(f"no run store at {src_root}")
    if src_root.resolve() == dst_store.root.resolve():
        raise ValueError("source and destination are the same store")

    with dst_store._connect() as conn:
        conn.execute("ATTACH DATABASE ? AS src", (str(src_db),))
        version = conn.execute("SELECT value FROM src.meta WHERE key = 'schema_version'").fetchone()
        if version is None or int(version[0]) != SCHEMA_VERSION:
            raise ValueError(f"source schema version {version} != {SCHEMA_VERSION}")
        run_ids = [r[0] for r in conn.execute("SELECT id FROM src.runs ORDER BY started_at")]
        clashes = [
            rid
            for rid in run_ids
            if conn.execute("SELECT 1 FROM runs WHERE id = ?", (rid,)).fetchone()
            or any((dst_store.root / tree / rid).exists() for tree in _RUN_FILE_TREES)
        ]
        if clashes:
            raise ValueError(f"runs already in destination: {clashes}")
        # Files first: if a copy fails, the transaction below never runs and the DB is unchanged.
        copied: list[Path] = []
        try:
            for rid in run_ids:
                for tree in _RUN_FILE_TREES:
                    src_dir = src_root / tree / rid
                    if src_dir.is_dir():
                        copied.append(dst_store.root / tree / rid)
                        shutil.copytree(src_dir, copied[-1])
            conn.execute("INSERT INTO runs SELECT * FROM src.runs")
            conn.execute("INSERT INTO metrics SELECT * FROM src.metrics")
        except BaseException:
            conn.rollback()
            for path in copied:
                shutil.rmtree(path, ignore_errors=True)
            raise
    return run_ids


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
