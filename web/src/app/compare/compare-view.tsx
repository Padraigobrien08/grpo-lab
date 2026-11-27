"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Line, XAxis, YAxis } from "recharts";

import { ErrorBlock, LoadingBlock } from "@/components/states";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { type ChartConfig, ChartContainer, ChartTooltip } from "@/components/ui/chart";
import { fetchJson, type Metrics, paths, type RunSummary, useJson } from "@/lib/api";
import { COMPARE_METRICS, groupCurve, groupRuns, type RunGroup, seedStat } from "@/lib/compare";
import { formatMetric, formatTick, isFraction, niceTicks, stepTicks } from "@/lib/metrics";
import { cn } from "@/lib/utils";

// Heavier than the run page's default: comparing seeds is about the trend, not step noise.
const SMOOTHING = 0.8;
const color = (g: RunGroup) => `var(--chart-${(g.colorIndex % 5) + 1})`;

/** Metrics for every run id given; null until all have loaded. */
function useAllMetrics(ids: string[]): { data: Record<string, Metrics> | null; error: string | null } {
  const key = ids.join("|");
  const [state, setState] = useState<{ key: string; data: Record<string, Metrics> | null; error: string | null }>(
    { key: "", data: null, error: null },
  );
  useEffect(() => {
    let live = true;
    const list = key ? key.split("|") : [];
    Promise.all(list.map((id) => fetchJson<Metrics>(paths.metrics(id)))).then(
      (all) => live && setState({ key, data: Object.fromEntries(list.map((id, i) => [id, all[i]])), error: null }),
      (err: unknown) => live && setState({ key, data: null, error: String(err) }),
    );
    return () => {
      live = false;
    };
  }, [key]);
  return state.key === key ? state : { data: null, error: null };
}

export function CompareView() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const runs = useJson<RunSummary[]>(paths.runs());
  const groups = runs.state === "ready" ? groupRuns(runs.data) : [];
  const showSingles = params.get("singles") === "1";
  const visible = groups.filter((g) => showSingles || g.runs.length > 1 || g.isReference);
  const metrics = useAllMetrics(visible.flatMap((g) => g.runs.map((r) => r.id)));

  const set = (k: string, v: string | null) => {
    const q = new URLSearchParams(params.toString());
    if (v === null) q.delete(k);
    else q.set(k, v);
    router.replace(`${pathname}?${q.toString()}`, { scroll: false });
  };

  if (runs.state === "error") return <ErrorBlock error={runs.error} />;
  if (metrics.error) return <ErrorBlock error={metrics.error} />;
  if (runs.state !== "ready" || !metrics.data) return <LoadingBlock rows={6} />;

  const logged = new Set(Object.values(metrics.data).flatMap((m) => Object.keys(m)));
  const options = COMPARE_METRICS.filter((m) => logged.has(m.key));
  const requested = params.get("metric");
  const metric = options.find((m) => m.key === requested) ?? options[0];
  const singles = groups.length - groups.filter((g) => g.runs.length > 1 || g.isReference).length;

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div
          role="radiogroup"
          aria-label="Metric"
          className="inline-flex w-fit flex-wrap rounded-lg border bg-muted/40 p-0.5 text-xs"
        >
          {options.map((m) => (
            <button
              key={m.key}
              type="button"
              role="radio"
              aria-checked={m.key === metric?.key}
              title={m.hint}
              onClick={() => set("metric", m.key === options[0].key ? null : m.key)}
              className={cn(
                "rounded-md px-2.5 py-1 font-medium text-muted-foreground transition-colors hover:text-foreground",
                m.key === metric?.key && "bg-background text-foreground shadow-xs",
              )}
            >
              {m.label}
            </button>
          ))}
        </div>
        {singles > 0 && (
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={showSingles}
              onChange={(e) => set("singles", e.target.checked ? "1" : null)}
            />
            Show {singles} single-run {singles === 1 ? "config" : "configs"} (e.g. pilots)
          </label>
        )}
      </div>
      {metric ? (
        <>
          <BandChart groups={visible} metrics={metrics.data} metricKey={metric.key} hint={metric.hint} />
          <GroupTable groups={visible} metrics={metrics.data} metricKey={metric.key} />
        </>
      ) : (
        <p className="text-sm text-muted-foreground">These runs logged none of the comparable metrics.</p>
      )}
    </>
  );
}

function BandChart({
  groups,
  metrics,
  metricKey,
  hint,
}: {
  groups: RunGroup[];
  metrics: Record<string, Metrics>;
  metricKey: string;
  hint: string;
}) {
  const curves = groups.map((g) => ({
    g,
    id: `g${g.colorIndex}`,
    points: groupCurve(
      g.runs.map((r) => metrics[r.id]?.[metricKey] ?? []).filter((s) => s.length > 0),
      SMOOTHING,
    ),
  }));
  const config: ChartConfig = Object.fromEntries(
    curves.map((c) => [c.id, { label: c.g.label, color: color(c.g) }]),
  );
  const rows = new Map<number, Record<string, unknown>>();
  for (const c of curves) {
    for (const { step, stat } of c.points) {
      const row = rows.get(step) ?? { step };
      row[c.id] = stat.mean;
      row[`${c.id}_band`] = [stat.lo, stat.hi];
      row[`${c.id}_n`] = stat.n;
      rows.set(step, row);
    }
  }
  const data = [...rows.values()].sort((a, b) => (a.step as number) - (b.step as number));
  const lastStep = (data.at(-1)?.step as number | undefined) ?? 0;
  const drawn = curves.flatMap((c) => c.points.flatMap((p) => [p.stat.lo, p.stat.hi]));
  let yTicks = niceTicks(Math.min(...drawn), Math.max(...drawn));
  if (isFraction(metricKey)) yTicks = yTicks.filter((t) => t >= 0 && t <= 1);

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>{COMPARE_METRICS.find((m) => m.key === metricKey)?.label}</CardTitle>
        <CardDescription>
          {hint}. Line: mean across seeds; band: 95% t-interval (needs ≥ 2 seeds, and is wide with
          only 2). Training metrics are EMA-smoothed per run before averaging.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <ChartContainer config={config} className="aspect-auto h-80 w-full">
          <ComposedChart data={data} margin={{ left: 4, right: 12, top: 4 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="step" type="number" domain={[0, lastStep]} ticks={stepTicks(lastStep)} tickLine={false} />
            <YAxis
              width={48}
              tickLine={false}
              axisLine={false}
              domain={[yTicks[0], yTicks.at(-1) ?? 1]}
              ticks={yTicks}
              tickFormatter={(v: number) => formatTick(v, metricKey)}
            />
            <ChartTooltip
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null;
                const row = payload[0].payload as Record<string, unknown>;
                return (
                  <div className="grid min-w-44 gap-1.5 rounded-lg border bg-background px-2.5 py-1.5 text-xs shadow-xl">
                    <span className="font-medium">step {label}</span>
                    {curves
                      .filter((c) => row[c.id] !== undefined)
                      .map((c) => {
                        const [lo, hi] = row[`${c.id}_band`] as [number, number];
                        const n = row[`${c.id}_n`] as number;
                        return (
                          <div key={c.id} className="flex items-center gap-2">
                            <span className="size-2 shrink-0 rounded-[2px]" style={{ background: color(c.g) }} />
                            <span className="text-muted-foreground">{c.g.label}</span>
                            <span className="ml-auto font-mono tabular-nums">
                              {formatMetric(row[c.id] as number, metricKey)}
                              {n > 1 && (
                                <span className="text-muted-foreground">
                                  {" "}
                                  [{formatMetric(lo, metricKey)}, {formatMetric(hi, metricKey)}]
                                </span>
                              )}
                            </span>
                          </div>
                        );
                      })}
                  </div>
                );
              }}
            />
            {curves.map((c) => (
              <Area
                key={`${c.id}_band`}
                dataKey={`${c.id}_band`}
                fill={color(c.g)}
                fillOpacity={0.14}
                stroke="none"
                isAnimationActive={false}
                connectNulls
                activeDot={false}
              />
            ))}
            {curves.map((c) => (
              <Line
                key={c.id}
                dataKey={c.id}
                stroke={color(c.g)}
                strokeWidth={2}
                dot={c.points.length < 50 ? { r: 2.5 } : false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </ComposedChart>
        </ChartContainer>
        {/* the legend is always present for >= 2 groups; identity never rests on colour alone */}
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Groups">
          {curves.map((c) => (
            <li key={c.id} className="flex items-center gap-1.5">
              <span className="h-0.5 w-4 rounded-full" style={{ background: color(c.g) }} aria-hidden />
              <span>{c.g.label}</span>
              <span className="text-muted-foreground">
                · {c.g.runs.length} {c.g.runs.length === 1 ? "run" : "seeds"}
              </span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function GroupTable({
  groups,
  metrics,
  metricKey,
}: {
  groups: RunGroup[];
  metrics: Record<string, Metrics>;
  metricKey: string;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border">
      <table className="w-full text-sm tabular-nums">
        <thead className="border-b text-left text-xs text-muted-foreground">
          <tr>
            <th className="px-3 py-2 font-medium">Group</th>
            <th className="px-3 py-2 font-medium">Differs from baseline</th>
            <th className="px-3 py-2 text-right font-medium">Final (mean, 95% CI)</th>
            <th className="px-3 py-2 font-medium">Runs</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => {
            const finals = g.runs
              .map((r) => metrics[r.id]?.[metricKey]?.at(-1)?.[1])
              .filter((v): v is number => v !== undefined);
            const s = finals.length ? seedStat(finals) : null;
            return (
              <tr key={g.key} className="border-b last:border-0 align-top">
                <td className="px-3 py-2">
                  <span className="flex items-center gap-2 font-medium">
                    <span className="size-2.5 shrink-0 rounded-[3px]" style={{ background: color(g) }} aria-hidden />
                    {g.label}
                  </span>
                </td>
                <td className="px-3 py-2">
                  {g.isReference ? (
                    <Badge variant="secondary">baseline</Badge>
                  ) : (
                    <span className="flex flex-wrap gap-1">
                      {g.diff.map(([k, v]) => (
                        <code key={k} className="rounded bg-muted px-1.5 py-0.5 text-xs">
                          {k}={v}
                        </code>
                      ))}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right font-mono whitespace-nowrap">
                  {s ? formatMetric(s.mean, metricKey) : "—"}
                  {s && s.n > 1 && (
                    <span className="text-muted-foreground">
                      {" "}
                      [{formatMetric(s.lo, metricKey)}, {formatMetric(s.hi, metricKey)}]
                    </span>
                  )}
                </td>
                <td className="px-3 py-2">
                  <span className="flex flex-wrap gap-x-2 gap-y-0.5 text-xs">
                    {g.runs.map((r) => (
                      <Link
                        key={r.id}
                        href={`/run?id=${encodeURIComponent(r.id)}`}
                        className="text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                      >
                        seed {r.seed ?? "?"}
                      </Link>
                    ))}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
