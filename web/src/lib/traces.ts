import type { TraceRow } from "@/lib/api";

/** What the token heatmap colours by. "text" is the normal rendered completion. */
export type TokenView = "text" | "surprisal" | "entropy" | "kl" | "ratio";

export const TOKEN_VIEWS: { id: TokenView; label: string; hint: string }[] = [
  { id: "text", label: "Text", hint: "The completion, with math rendered" },
  {
    id: "surprisal",
    label: "Surprisal",
    hint: "−log p of each sampled token under the sampling policy: darker = less likely",
  },
  { id: "entropy", label: "Entropy", hint: "Entropy of the policy at each position: darker = less certain" },
  { id: "kl", label: "KL", hint: "Per-token k3 KL to the reference model: darker = drifted further" },
  {
    id: "ratio",
    label: "Ratio",
    hint: "π / π_old on the last update pass: blue above 1, orange below",
  },
];

export function isTokenView(v: string | null): v is TokenView {
  return TOKEN_VIEWS.some((t) => t.id === v);
}

/** The per-token values a view colours by, or null when the trace doesn't have them. */
export function viewValues(row: TraceRow, view: TokenView): number[] | null {
  switch (view) {
    case "surprisal":
      return row.logprob.map((x) => -x);
    case "entropy":
      return row.entropy;
    case "kl":
      return row.kl;
    case "ratio":
      return row.ratio;
    default:
      return null;
  }
}

/** Views with nothing to show for these traces: no reference model, or a single pass. */
export function unavailableViews(rows: TraceRow[]): Partial<Record<TokenView, string>> {
  const out: Partial<Record<TokenView, string>> = {};
  if (rows.every((r) => r.kl === null)) out.kl = "No reference model in this run (β = 0)";
  if (rows.every((r) => r.ratio.every((x) => x === 1))) {
    out.ratio = "Exactly 1 everywhere: one update pass per batch (μ = 1)";
  }
  return out;
}

export type Scale = {
  diverging: boolean;
  /** values at or beyond this saturate; the 95th percentile, so one outlier can't wash out the rest */
  max: number;
  /** 0..1 strength and which side of the midpoint, for one value */
  at: (v: number) => { strength: number; above: boolean };
};

/**
 * One scale per group, shared by all its samples, so siblings can be compared token by token.
 * Sequential views run from 0; ratio is diverging around 1 on a log scale.
 */
export function groupScale(rows: TraceRow[], view: TokenView): Scale | null {
  const all = rows.flatMap((r) => viewValues(r, view) ?? []);
  if (all.length === 0) return null;
  const diverging = view === "ratio";
  const mags = all.map((v) => (diverging ? Math.abs(Math.log(v)) : Math.max(v, 0)));
  const max = Math.max(percentile(mags, 0.95), 1e-6);
  return {
    diverging,
    max: diverging ? Math.exp(max) : max,
    at: (v) => {
      const m = diverging ? Math.abs(Math.log(v)) : Math.max(v, 0);
      return { strength: Math.min(1, m / max), above: diverging ? v >= 1 : true };
    },
  };
}

function percentile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
}

export function formatValue(v: number): string {
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 1000)) return v.toExponential(1);
  return v.toFixed(a < 1 ? 3 : 2);
}

/** Tooltip text for one token: every recorded value, whichever view is showing. */
export function tokenTitle(row: TraceRow, i: number): string {
  const lines = [
    JSON.stringify(row.tokens[i]),
    `log p ${formatValue(row.logprob[i])} (p = ${formatValue(Math.exp(row.logprob[i]))})`,
    `entropy ${formatValue(row.entropy[i])}`,
  ];
  if (row.kl) lines.push(`KL ${formatValue(row.kl[i])}`);
  lines.push(`ratio ${formatValue(row.ratio[i])}`);
  return lines.join("\n");
}
