import type { Metrics } from "@/lib/api";

const last = (m: Metrics, k: string) => m[k]?.at(-1)?.[1];
const first = (m: Metrics, k: string) => m[k]?.[0]?.[1];

/** One line of a run's headline numbers, at data size: what changed, start to finish. */
export function RunSummary({ metrics }: { metrics: Metrics }) {
  const acc0 = first(metrics, "eval/accuracy");
  const acc1 = last(metrics, "eval/accuracy");
  const items: { label: string; value: React.ReactNode }[] = [];
  if (acc0 !== undefined && acc1 !== undefined) {
    const delta = (acc1 - acc0) * 100;
    items.push({
      label: "Eval accuracy",
      value: (
        <>
          {(acc0 * 100).toFixed(1)}% → <strong>{(acc1 * 100).toFixed(1)}%</strong>{" "}
          <span className={delta >= 0 ? "text-correct" : "text-incorrect"}>
            ({delta >= 0 ? "+" : "−"}
            {Math.abs(delta).toFixed(1)} pts)
          </span>
        </>
      ),
    });
  }
  const stat = (label: string, key: string, fmt: (v: number) => string) => {
    const v = last(metrics, key);
    if (v !== undefined) items.push({ label, value: fmt(v) });
  };
  stat("Final reward", "reward/mean", (v) => v.toFixed(3));
  stat("Final KL", "kl", (v) => v.toFixed(4));
  stat("Zero-advantage groups", "adv/zero_group_frac", (v) => `${(v * 100).toFixed(0)}%`);
  if (items.length === 0) return null;

  return (
    <dl className="flex flex-wrap gap-x-8 gap-y-3 tabular-nums">
      {items.map((it) => (
        <div key={it.label} className="flex flex-col gap-0.5">
          <dt className="text-xs text-muted-foreground">{it.label}</dt>
          <dd className="text-sm">{it.value}</dd>
        </div>
      ))}
    </dl>
  );
}
