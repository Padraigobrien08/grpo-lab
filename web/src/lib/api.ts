"use client";

import { useEffect, useState } from "react";

// Mirrors the FastAPI responses in api/src/grpo_api/main.py.
export type RunSummary = {
  id: string;
  name: string;
  config: Record<string, unknown>;
  git_sha: string | null;
  seed: number | null;
  status: string;
  started_at: number;
  finished_at: number | null;
  last_step: number | null;
  last_metrics: Record<string, number>;
  rollout_steps: number;
};

export type RunDetail = Omit<RunSummary, "last_step" | "last_metrics" | "rollout_steps"> & {
  metric_keys: string[];
};

/** metric key -> [step, value] pairs, sorted by step */
export type Metrics = Record<string, [number, number][]>;

export type RolloutRow = {
  run_id: string;
  step: number;
  prompt_id: string;
  group_idx: number;
  sample_idx: number;
  prompt: string;
  completion: string;
  reward_total: number;
  reward_components: string; // JSON object
  gold_reward: number | null;
  advantage: number;
  length: number;
  truncated: boolean;
};

/** Per-token values for one traced sample; every list has one entry per completion token. */
export type TraceRow = {
  prompt_id: string;
  group_idx: number;
  sample_idx: number;
  token_ids: number[];
  tokens: string[];
  logprob: number[];
  ref_logprob: number[] | null; // null when the run had no reference model (beta = 0)
  kl: number[] | null;
  entropy: number[];
  ratio: number[];
};

/**
 * Two data sources behind the same paths:
 * - NEXT_PUBLIC_API_URL set: the live FastAPI server (local development against a run store).
 * - unset: the static snapshot that `uv run export-static` writes to public/data/<path>.json.
 */
const API_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");

export function dataUrl(path: string): string {
  return API_URL ? `${API_URL}${path}` : `/data${path}.json`;
}

export const paths = {
  runs: () => "/runs",
  run: (id: string) => `/runs/${encodeURIComponent(id)}`,
  metrics: (id: string) => `/runs/${encodeURIComponent(id)}/metrics`,
  rolloutSteps: (id: string) => `/runs/${encodeURIComponent(id)}/rollouts`,
  rollouts: (id: string, step: number) => `/runs/${encodeURIComponent(id)}/rollouts/${step}`,
  traceSteps: (id: string) => `/runs/${encodeURIComponent(id)}/traces`,
  traceGroups: (id: string, step: number) => `/runs/${encodeURIComponent(id)}/traces/${step}`,
  traces: (id: string, step: number, group: number) =>
    `/runs/${encodeURIComponent(id)}/traces/${step}/${group}`,
};

const cache = new Map<string, Promise<unknown>>();

export function fetchJson<T>(path: string): Promise<T> {
  const url = dataUrl(path);
  let hit = cache.get(url);
  if (!hit) {
    hit = fetch(url).then((res) => {
      if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
      return res.json();
    });
    hit.catch(() => cache.delete(url)); // let a failed request be retried
    cache.set(url, hit);
  }
  return hit as Promise<T>;
}

export type Loadable<T> =
  | { state: "loading" }
  | { state: "error"; error: string }
  | { state: "ready"; data: T };

/** Fetch `path` (or nothing when null) and track loading/error state. */
export function useJson<T>(path: string | null): Loadable<T> {
  const [result, setResult] = useState<{ path: string | null; value: Loadable<T> }>({
    path: null,
    value: { state: "loading" },
  });
  useEffect(() => {
    if (path === null) return;
    let live = true;
    fetchJson<T>(path).then(
      (data) => live && setResult({ path, value: { state: "ready", data } }),
      (err: unknown) =>
        live && setResult({ path, value: { state: "error", error: String(err) } }),
    );
    return () => {
      live = false;
    };
  }, [path]);
  // A result for an earlier path is stale: show loading until the new one arrives.
  return result.path === path ? result.value : { state: "loading" };
}
