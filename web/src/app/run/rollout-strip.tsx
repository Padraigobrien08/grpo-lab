import Link from "next/link";

/**
 * The logged rollout steps as ticks on the run's step axis. Each tick opens that step in the
 * inspector; the strip reads as "where along training can I look at samples".
 */
export function RolloutStrip({
  runId,
  steps,
  lastStep,
}: {
  runId: string;
  steps: number[];
  lastStep: number;
}) {
  if (steps.length === 0) {
    return <p className="text-xs text-muted-foreground">This run logged no rollouts.</p>;
  }
  const span = Math.max(lastStep, steps.at(-1) ?? 1);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-4 text-xs text-muted-foreground">
        <span>
          Rollout groups logged at {steps.length} steps. Pick one to inspect its samples.
        </span>
        <span className="whitespace-nowrap tabular-nums">0–{span}</span>
      </div>
      <div className="relative h-7 rounded-md bg-muted/60" role="list" aria-label="Logged steps">
        {steps.map((s) => (
          <Link
            key={s}
            role="listitem"
            href={`/rollouts?id=${encodeURIComponent(runId)}&step=${s}`}
            title={`Step ${s}`}
            aria-label={`Rollouts at step ${s}`}
            style={{ left: `${(s / span) * 100}%` }}
            className="group absolute inset-y-0 flex w-3 -translate-x-1/2 justify-center rounded-sm focus-visible:outline-offset-0"
          >
            <span className="my-1.5 w-px bg-foreground/40 transition-colors group-hover:w-0.5 group-hover:bg-foreground group-focus-visible:bg-foreground" />
          </Link>
        ))}
      </div>
    </div>
  );
}
