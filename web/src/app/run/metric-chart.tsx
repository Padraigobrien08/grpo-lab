"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  type ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { Metrics } from "@/lib/api";
import { ema, formatMetric, formatTick, isFraction, niceTicks, stepTicks } from "@/lib/metrics";

// Series with fewer points than this (e.g. eval, every 25 steps) are drawn raw: smoothing a
// dozen points mostly just lags them.
const SMOOTH_MIN_POINTS = 50;

type Props = {
  title: string;
  description: string;
  keys: string[];
  metrics: Metrics;
  smoothing: number;
  onPickStep: (step: number) => void;
};

export function MetricChart({ title, description, keys, metrics, smoothing, onPickStep }: Props) {
  // Metric keys contain "/", which can't appear in the CSS variables the chart uses for color.
  const series = keys.map((key, i) => ({ key, id: `s${i}`, points: metrics[key] ?? [] }));
  const config: ChartConfig = Object.fromEntries(
    series.map((s, i) => [s.id, { label: s.key, color: `var(--chart-${(i % 5) + 1})` }]),
  );

  const rows = new Map<number, Record<string, number>>();
  for (const s of series) {
    const raw = s.points.map(([, v]) => v);
    const smooth = s.points.length >= SMOOTH_MIN_POINTS ? ema(raw, smoothing) : raw;
    s.points.forEach(([step], j) => {
      const row = rows.get(step) ?? { step };
      row[s.id] = smooth[j];
      row[`${s.id}_raw`] = raw[j];
      rows.set(step, row);
    });
  }
  const data = [...rows.values()].sort((a, b) => a.step - b.step);
  const showRaw = smoothing > 0;
  const lastStep = data.at(-1)?.step ?? 0;
  // Round ticks over everything drawn (raw lines included); fractions stay within [0, 1].
  const drawn = data.flatMap((row) =>
    Object.entries(row)
      .filter(([k]) => k !== "step" && (showRaw || !k.endsWith("_raw")))
      .map(([, v]) => v),
  );
  let yTicks = niceTicks(Math.min(...drawn), Math.max(...drawn));
  if (keys.every(isFraction)) yTicks = yTicks.filter((t) => t >= 0 && t <= 1);

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="aspect-auto h-56 w-full">
          <LineChart
            data={data}
            margin={{ left: 4, right: 12, top: 4 }}
            onClick={(state) => {
              const step = Number(state?.activeLabel);
              if (Number.isFinite(step)) onPickStep(step);
            }}
            className="cursor-pointer"
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="step"
              type="number"
              domain={[0, lastStep]}
              ticks={stepTicks(lastStep)}
              tickLine={false}
            />
            <YAxis
              width={48}
              tickLine={false}
              axisLine={false}
              domain={[yTicks[0], yTicks.at(-1) ?? 1]}
              ticks={yTicks}
              tickFormatter={(v: number) => formatTick(v, keys[0])}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(_, payload) => `step ${payload?.[0]?.payload?.step ?? ""}`}
                  formatter={(value, name) => (
                    <div className="flex w-full justify-between gap-4">
                      <span className="text-muted-foreground">{config[String(name)]?.label}</span>
                      <span className="font-mono tabular-nums">
                        {formatMetric(Number(value), keys[Number(String(name).slice(1))])}
                      </span>
                    </div>
                  )}
                />
              }
            />
            {showRaw &&
              series
                .filter((s) => s.points.length >= SMOOTH_MIN_POINTS)
                .map((s) => (
                  <Line
                    key={`${s.id}_raw`}
                    dataKey={`${s.id}_raw`}
                    stroke={`var(--color-${s.id})`}
                    strokeOpacity={0.2}
                    dot={false}
                    isAnimationActive={false}
                    tooltipType="none"
                    legendType="none"
                    connectNulls
                  />
                ))}
            {series.map((s) => (
              <Line
                key={s.id}
                dataKey={s.id}
                stroke={`var(--color-${s.id})`}
                strokeWidth={1.75}
                dot={s.points.length < SMOOTH_MIN_POINTS ? { r: 2 } : false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
            {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
