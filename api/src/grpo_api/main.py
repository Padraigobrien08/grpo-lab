"""Read-only API over the run store. Run with ``uv run uvicorn grpo_api.main:app --reload``."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from runstore import RunStore


def create_app(store_dir: str | None = None) -> FastAPI:
    store = RunStore(store_dir or os.environ.get("GRPO_STORE", "runs"))
    app = FastAPI(title="grpo-lab API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("GRPO_CORS_ORIGINS", "http://localhost:3000").split(","),
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def _require_run(run_id: str) -> None:
        if store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        # Summaries ride along so the runs table needs one request, not one per run.
        out = []
        for run in store.list_runs():
            last = store.last_metrics(run.id)
            out.append(
                {
                    **asdict(run),
                    "last_step": max((step for step, _ in last.values()), default=None),
                    "last_metrics": {key: value for key, (_, value) in last.items()},
                    "rollout_steps": len(store.rollout_steps(run.id)),
                }
            )
        return out

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return {**asdict(run), "metric_keys": store.metric_keys(run_id)}

    @app.get("/runs/{run_id}/metrics")
    def get_metrics(
        run_id: str, keys: Annotated[list[str] | None, Query()] = None
    ) -> dict[str, list[tuple[int, float]]]:
        _require_run(run_id)
        return store.get_metrics(run_id, keys)

    @app.get("/runs/{run_id}/rollouts")
    def rollout_steps(run_id: str) -> list[int]:
        _require_run(run_id)
        return store.rollout_steps(run_id)

    @app.get("/runs/{run_id}/rollouts/{step}")
    def rollouts(run_id: str, step: int) -> list[dict[str, Any]]:
        _require_run(run_id)
        return store.read_rollouts(run_id, step)

    return app


app = create_app()
