import type { RunSummary } from "@/lib/api";
import { flatten } from "@/lib/config-diff";
import { ema } from "@/lib/metrics";

/**
 * Config keys that don't change what a run trains: which seed, what it's called, where it ran
 * and what it logged. Runs that differ only in these are seeds of the same experiment.
 */
export function isNuisanceKey(key: string): boolean {
  return key === "name" || key === "seed" || key === "device" || key.startsWith("log.");
}

export type RunGroup = {
  key: string;
  /** Run name with the seed suffix removed, e.g. "m7-drgrpo" from "m7-drgrpo-s1". */
  label: string;
  runs: RunSummary[];
  /** [key, value] pairs where this group's config departs from the reference group's. */
  diff: [string, string][];
  isReference: boolean;
  /** Fixed per group across filters, so a group keeps its colour when others are hidden. */
  colorIndex: number;
};

function signature(config: Record<string, unknown>): Record<string, string> {
  return Object.fromEntries(Object.entries(flatten(config)).filter(([k]) => !isNuisanceKey(k)));
}

function baseName(name: string): string {
  return name.replace(/-(s|seed)\d+$/, "");
}

/**
 * Runs grouped by their training config, ignoring nuisance keys. The reference is the group
 * with the most runs (ties: the earliest), and every group is labelled by how it differs.
 */
export function groupRuns(runs: RunSummary[]): RunGroup[] {
  const byKey = new Map<string, { sig: Record<string, string>; runs: RunSummary[] }>();
  for (const run of runs) {
    const sig = signature(run.config);
    const key = JSON.stringify(Object.entries(sig).sort(([a], [b]) => a.localeCompare(b)));
    const entry = byKey.get(key) ?? { sig, runs: [] };
    entry.runs.push(run);
    byKey.set(key, entry);
  }
  const firstStart = (rs: RunSummary[]) => Math.min(...rs.map((r) => r.started_at));
  // Colour order: oldest experiment first, so adding runs never repaints existing groups.
  const entries = [...byKey.entries()].sort(([, a], [, b]) => firstStart(a.runs) - firstStart(b.runs));
  const ref = [...entries].sort(
    ([, a], [, b]) => b.runs.length - a.runs.length || firstStart(a.runs) - firstStart(b.runs),
  )[0];
  const groups = entries.map(([key, { sig, runs: members }], i): RunGroup => {
    const sorted = [...members].sort((a, b) => (a.seed ?? 0) - (b.seed ?? 0));
    const keys = [...new Set([...Object.keys(sig), ...Object.keys(ref[1].sig)])].sort();
    return {
      key,
      label: baseName(sorted[0].name),
      runs: sorted,
      diff: keys.filter((k) => sig[k] !== ref[1].sig[k]).map((k) => [k, sig[k] ?? "unset"]),
      isReference: key === ref[0],
      colorIndex: i,
    };
  });
  // Listed baseline first; colours were fixed above by age, so this reorders nothing visual.
  return [...groups.filter((g) => g.isReference), ...groups.filter((g) => !g.isReference)];
}

// Two-sided 95% Student-t critical values by degrees of freedom (df >= 30 uses ~normal).
const T95 = [NaN, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228];
function t95(df: number): number {
  if (df < 1) return NaN;
  if (df < T95.length) return T95[df];
  return df < 30 ? 2.1 : 1.96;
}

export type SeedStat = { mean: number; lo: number; hi: number; n: number };

/** Mean and 95% t-interval across seeds; with one run, the interval collapses to the value. */
export function seedStat(values: number[]): SeedStat {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, lo: mean, hi: mean, n };
  const sd = Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1));
  const half = (t95(n - 1) * sd) / Math.sqrt(n);
  return { mean, lo: mean - half, hi: mean + half, n };
}

// Same rule as the run page: only long series are smoothed (eval every 25 steps is not).
const SMOOTH_MIN_POINTS = 50;

/**
 * Per-step seed statistics for one group, over the steps every run logged. Each run is smoothed
 * on its own first, so the band shows spread between seeds rather than step-to-step noise.
 */
export function groupCurve(
  series: [number, number][][],
  smoothing: number,
): { step: number; stat: SeedStat }[] {
  const perRun = series.map((points) => {
    const values = points.map(([, v]) => v);
    const smooth = points.length >= SMOOTH_MIN_POINTS ? ema(values, smoothing) : values;
    return new Map(points.map(([step], i) => [step, smooth[i]]));
  });
  if (perRun.length === 0) return [];
  const steps = [...perRun[0].keys()].filter((s) => perRun.every((m) => m.has(s)));
  return steps
    .sort((a, b) => a - b)
    .map((step) => ({ step, stat: seedStat(perRun.map((m) => m.get(step) as number)) }));
}

/** Metrics the compare page offers, in order. Keys no run logged are hidden. */
export const COMPARE_METRICS: { key: string; label: string; hint: string }[] = [
  { key: "eval/accuracy", label: "Eval accuracy", hint: "Greedy, on the fixed 500-problem test subset" },
  { key: "reward/mean", label: "Reward", hint: "Mean training reward" },
  { key: "completion/len_mean", label: "Length", hint: "Tokens per training completion" },
  { key: "entropy", label: "Entropy", hint: "Policy entropy per token" },
  { key: "kl", label: "KL", hint: "k3 KL to the reference, per token (absent when β = 0)" },
  {
    key: "adv/zero_group_frac",
    label: "Zero-advantage",
    hint: "Share of groups where every sample got the same reward",
  },
];
