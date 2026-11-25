"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { ErrorBlock, LoadingBlock } from "@/components/states";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { type Metrics, paths, type RunDetail, useJson } from "@/lib/api";
import { groupMetrics } from "@/lib/metrics";

import { MetricChart } from "./metric-chart";

export function RunDetailView() {
  const id = useSearchParams().get("id");
  const run = useJson<RunDetail>(id && paths.run(id));
  const metrics = useJson<Metrics>(id && paths.metrics(id));
  const steps = useJson<number[]>(id && paths.rolloutSteps(id));
  const [smoothing, setSmoothing] = useState(0.6);
  const router = useRouter();

  if (!id) return <ErrorBlock error="No run id in the URL (expected /run?id=...)" />;
  for (const r of [run, metrics, steps]) if (r.state === "error") return <ErrorBlock error={r.error} />;
  if (run.state !== "ready" || metrics.state !== "ready" || steps.state !== "ready") {
    return <LoadingBlock rows={6} />;
  }

  const rolloutSteps = steps.data;
  const inspect = (step: number) => {
    // jump to the logged rollout step closest to the one clicked
    if (rolloutSteps.length === 0) return;
    const nearest = rolloutSteps.reduce((a, b) => (Math.abs(b - step) < Math.abs(a - step) ? b : a));
    router.push(`/rollouts?id=${encodeURIComponent(id)}&step=${nearest}`);
  };

  return (
    <>
      <header className="flex flex-col gap-3">
        <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
          ← Runs
        </Link>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">{run.data.name}</h1>
          <Badge variant="secondary">{run.data.status}</Badge>
          {run.data.seed !== null && <Badge variant="outline">seed {run.data.seed}</Badge>}
        </div>
        <p className="font-mono text-xs text-muted-foreground">
          {run.data.id}
          {run.data.git_sha && ` · ${run.data.git_sha.slice(0, 7)}`}
        </p>
        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            Config
          </summary>
          <pre className="mt-2 max-h-96 overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-xs">
            {JSON.stringify(run.data.config, null, 2)}
          </pre>
        </details>
      </header>

      <section className="flex flex-col gap-3 rounded-lg border p-4">
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex min-w-64 items-center gap-3 text-sm">
            <span className="whitespace-nowrap text-muted-foreground">Smoothing</span>
            <Slider
              value={[smoothing]}
              min={0}
              max={0.99}
              step={0.01}
              onValueChange={(v) => setSmoothing(Array.isArray(v) ? v[0] : v)}
              aria-label="Smoothing"
            />
            <span className="w-10 text-right tabular-nums">{smoothing.toFixed(2)}</span>
          </label>
          <p className="text-xs text-muted-foreground">
            EMA over per-step metrics (faint line = raw). Click a chart to open the rollout
            groups logged nearest that step.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs text-muted-foreground">Rollouts logged at step</span>
          {rolloutSteps.length === 0 && <span className="text-xs">none</span>}
          {rolloutSteps.map((s) => (
            <Link
              key={s}
              href={`/rollouts?id=${encodeURIComponent(id)}&step=${s}`}
              className="rounded-md border px-1.5 py-0.5 font-mono text-xs tabular-nums hover:bg-muted"
            >
              {s}
            </Link>
          ))}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {groupMetrics(run.data.metric_keys).map((g) => (
          <MetricChart
            key={g.title}
            title={g.title}
            description={g.description}
            keys={g.keys}
            metrics={metrics.data}
            smoothing={smoothing}
            onPickStep={inspect}
          />
        ))}
      </section>
    </>
  );
}
