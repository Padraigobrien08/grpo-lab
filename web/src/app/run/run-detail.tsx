"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { BackLink } from "@/components/back-link";
import { ErrorBlock, RunSkeleton } from "@/components/states";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { type Metrics, paths, type RunDetail, useJson } from "@/lib/api";
import { groupMetrics } from "@/lib/metrics";

import { MetricChart } from "./metric-chart";
import { RolloutStrip } from "./rollout-strip";
import { RunSummary } from "./run-summary";

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
    return <RunSkeleton />;
  }

  const rolloutSteps = steps.data;
  const lastStep = Math.max(0, ...Object.values(metrics.data).map((pts) => pts.at(-1)?.[0] ?? 0));
  const nearest = (step: number) =>
    rolloutSteps.length === 0
      ? null
      : rolloutSteps.reduce((a, b) => (Math.abs(b - step) < Math.abs(a - step) ? b : a));
  const inspect = (step: number) => {
    const s = nearest(step);
    if (s !== null) router.push(`/rollouts?id=${encodeURIComponent(id)}&step=${s}`);
  };
  const groups = groupMetrics(run.data.metric_keys);
  const chart = (g: (typeof groups)[number], height: string) => (
    <MetricChart
      key={g.title}
      title={g.title}
      description={g.description}
      keys={g.keys}
      metrics={metrics.data}
      smoothing={smoothing}
      height={height}
      nearestRollout={nearest}
      onPickStep={inspect}
    />
  );

  return (
    <>
      <header className="flex flex-col gap-4">
        <BackLink href="/">Runs</BackLink>
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-semibold tracking-tight">{run.data.name}</h1>
            {run.data.seed !== null && <Badge variant="outline">seed {run.data.seed}</Badge>}
            {run.data.status !== "finished" && <Badge variant="secondary">{run.data.status}</Badge>}
          </div>
          <p className="font-mono text-xs text-muted-foreground">
            {run.data.id}
            {run.data.git_sha && ` · ${run.data.git_sha.slice(0, 7)}`}
          </p>
        </div>
        <RunSummary metrics={metrics.data} />
        <details className="text-sm">
          <summary className="w-fit cursor-pointer text-muted-foreground hover:text-foreground">
            Full config
          </summary>
          <pre className="mt-2 max-h-96 overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-xs">
            {JSON.stringify(run.data.config, null, 2)}
          </pre>
        </details>
      </header>

      <section className="flex flex-col gap-4 rounded-lg border p-4" aria-label="Chart controls">
        <RolloutStrip runId={id} steps={rolloutSteps} lastStep={lastStep} />
        <label className="flex max-w-md items-center gap-3 text-sm">
          <span className="whitespace-nowrap text-muted-foreground">Smoothing</span>
          <Slider
            value={[smoothing]}
            min={0}
            max={0.99}
            step={0.01}
            onValueChange={(v) => setSmoothing(Array.isArray(v) ? v[0] : v)}
            thumbLabel="Chart smoothing"
            valueText={(v) => `Smoothing ${v.toFixed(2)}${v === 0 ? " (raw values)" : ""}`}
          />
          <span className="w-10 text-right tabular-nums">{smoothing.toFixed(2)}</span>
        </label>
      </section>

      <section className="grid gap-4 md:grid-cols-2" aria-label="Key metrics">
        {groups.filter((g) => g.primary).map((g) => chart(g, "h-64"))}
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="diagnostics">
        <h2 id="diagnostics" className="text-base font-semibold">
          Diagnostics
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {groups.filter((g) => !g.primary).map((g) => chart(g, "h-44"))}
        </div>
      </section>
    </>
  );
}
