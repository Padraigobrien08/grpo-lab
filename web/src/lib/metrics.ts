/**
 * Exponential moving average with TensorBoard-style debiasing, so the start of a curve isn't
 * pulled toward zero. weight in [0, 1): 0 = raw values.
 */
export function ema(values: number[], weight: number): number[] {
  if (weight <= 0) return values;
  let last = 0;
  return values.map((v, i) => {
    last = last * weight + (1 - weight) * v;
    return last / (1 - weight ** (i + 1));
  });
}

export type MetricGroup = {
  title: string;
  description: string;
  keys: string[];
  /** The charts that tell the story of a run; shown first and larger. */
  primary?: boolean;
};

// Charts on the run page, in order. Keys a run doesn't have are skipped; unknown keys go to
// "Other", so a new metric shows up without a code change.
export const METRIC_GROUPS: MetricGroup[] = [
  {
    title: "Eval accuracy",
    primary: true,
    description: "Greedy accuracy on the fixed 500-problem test subset",
    keys: ["eval/accuracy"],
  },
  {
    title: "Reward",
    primary: true,
    description: "Mean training reward, and the share of samples answered correctly",
    keys: ["reward/mean", "reward/correct"],
  },
  { title: "KL to reference", description: "k3 estimator, per token", keys: ["kl"], primary: true },
  {
    title: "Zero-advantage groups",
    primary: true,
    description: "Groups where all samples got the same reward (no learning signal)",
    keys: ["adv/zero_group_frac"],
  },
  {
    title: "Boxed answers",
    description: "Share of completions with a \\boxed{} answer: training samples vs greedy eval",
    keys: ["reward/format", "eval/format"],
  },
  { title: "Entropy", description: "Policy entropy, per token", keys: ["entropy"] },
  {
    title: "Completion length",
    description: "Tokens per completion",
    keys: ["completion/len_mean", "completion/len_max"],
  },
  {
    title: "Truncation",
    description: "Fraction of completions that hit max_new_tokens",
    keys: ["completion/truncated_frac", "eval/truncated_frac"],
  },
  {
    title: "Clipping",
    description:
      "Share of tokens clipped by the ratio objective. Always 0 with one optimisation pass per batch (the ratio is exactly 1)",
    keys: ["clip_frac/low", "clip_frac/high"],
  },
  { title: "Reward spread", description: "Std of rewards within the batch", keys: ["reward/std"] },
  { title: "Gradient norm", description: "Before clipping", keys: ["grad_norm"] },
  { title: "Loss", description: "Policy loss incl. KL term", keys: ["loss"] },
  {
    title: "Time per step",
    description: "Seconds spent generating vs training",
    keys: ["time/gen_s", "time/train_s"],
  },
];

export function groupMetrics(keys: string[]): MetricGroup[] {
  // Story order for the primary charts: did it learn (eval), what it optimised (reward),
  // how far it moved (KL), and whether groups still carry signal (zero-advantage).
  const present = new Set(keys);
  const used = new Set<string>();
  const groups = METRIC_GROUPS.map((g) => {
    const k = g.keys.filter((key) => present.has(key));
    k.forEach((key) => used.add(key));
    return { ...g, keys: k };
  }).filter((g) => g.keys.length > 0);
  const rest = keys.filter((k) => !used.has(k));
  if (rest.length) {
    groups.push({ title: "Other", description: "Everything else the run logged", keys: rest });
  }
  return groups;
}

// Readable names for chart legends and tooltips; the raw key stays visible in tooltips.
const METRIC_LABELS: Record<string, string> = {
  "eval/accuracy": "Eval accuracy",
  "eval/format": "Eval: boxed answer",
  "eval/truncated_frac": "Eval: truncated",
  "eval/len_mean": "Eval: length",
  "reward/mean": "Mean reward",
  "reward/correct": "Share correct",
  "reward/format": "Training: boxed answer",
  "reward/std": "Reward std",
  "completion/len_mean": "Mean length",
  "completion/len_max": "Max length",
  "completion/truncated_frac": "Training: truncated",
  "clip_frac/low": "Clipped low",
  "clip_frac/high": "Clipped high",
  "adv/zero_group_frac": "Zero-advantage groups",
  "time/gen_s": "Generation",
  "time/train_s": "Training",
  "grad_norm": "Gradient norm",
  kl: "KL",
  entropy: "Entropy",
  loss: "Loss",
};

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? key;
}

/** Metrics that are fractions in [0, 1]: shown as percentages, axes clamped to the range. */
export function isFraction(key: string): boolean {
  return /accuracy|correct|format|frac/.test(key);
}

export function formatMetric(value: number | undefined, key = ""): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  if (isFraction(key)) return `${(value * 100).toFixed(1)}%`;
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toPrecision(3);
}

/** Short axis tick label: 3 significant digits, no trailing zeros (0.8, not 0.800). */
export function formatTick(value: number, key = ""): string {
  if (isFraction(key)) return `${Number((value * 100).toPrecision(3))}%`;
  return String(Number(value.toPrecision(3)));
}

/** Round ticks covering [lo, hi]; the first and last tick make a tidy axis domain. */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) [lo, hi] = [lo - Math.abs(lo || 1) * 0.1, hi + Math.abs(hi || 1) * 0.1];
  const raw = (hi - lo) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const size = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks = [];
  for (let t = Math.floor(lo / size) * size; t < hi + size / 2; t += size) {
    ticks.push(Number(t.toPrecision(12))); // strip float noise (0.30000000000000004)
  }
  return ticks;
}

/** Round step ticks (0, 50, 100, ...) from 0 to the last step. */
export function stepTicks(maxStep: number, target = 6): number[] {
  const raw = maxStep / target;
  const mag = 10 ** Math.floor(Math.log10(Math.max(raw, 1)));
  const size = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks = [];
  for (let t = 0; t <= maxStep; t += size) ticks.push(t);
  return ticks;
}
