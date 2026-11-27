"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

/**
 * The logged rollout steps as ticks on the run's step axis. Each tick opens that step in the
 * inspector; the strip reads as "where along training can I look at samples". On phones the
 * ticks would be ~10px apart, too close to tap, so a step picker replaces them there.
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
  const router = useRouter();
  if (steps.length === 0) {
    return <p className="text-xs text-muted-foreground">This run logged no rollouts.</p>;
  }
  const span = Math.max(lastStep, steps.at(-1) ?? 1);
  const href = (s: number) => `/rollouts?id=${encodeURIComponent(runId)}&step=${s}`;
  // Each tick's hit area is as wide as the smallest gap between logged steps, so neighbours
  // never overlap (steps 1 and 10 are closer than the rest), and never narrower than 12px.
  const minGap = Math.min(...steps.slice(1).map((s, i) => s - steps[i]), span);
  const hitWidth = `max(0.75rem, ${(minGap / span) * 100}%)`;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-4 text-xs text-muted-foreground">
        <span>
          Rollout groups logged at {steps.length} steps. Pick one to inspect its samples.
        </span>
        <span className="hidden whitespace-nowrap tabular-nums sm:inline">0–{span}</span>
      </div>
      <label className="flex items-center gap-2 text-sm sm:hidden">
        <span className="text-muted-foreground">Open step</span>
        <select
          className="h-10 min-w-0 flex-1 rounded-md border bg-background px-2 tabular-nums"
          defaultValue=""
          onChange={(e) => e.target.value && router.push(href(Number(e.target.value)))}
        >
          <option value="" disabled>
            Choose a logged step…
          </option>
          {steps.map((s) => (
            <option key={s} value={s}>
              Step {s}
            </option>
          ))}
        </select>
      </label>
      <div
        className="relative hidden h-7 rounded-md bg-muted/60 sm:block"
        role="list"
        aria-label="Logged steps"
      >
        {steps.map((s) => (
          <Link
            key={s}
            role="listitem"
            href={href(s)}
            title={`Step ${s}`}
            aria-label={`Rollouts at step ${s}`}
            style={{ left: `${(s / span) * 100}%`, width: hitWidth }}
            className="group absolute inset-y-0 flex -translate-x-1/2 justify-center rounded-sm focus-visible:outline-offset-0"
          >
            <span className="my-1.5 w-px bg-foreground/40 transition-colors group-hover:w-0.5 group-hover:bg-foreground group-focus-visible:bg-foreground" />
          </Link>
        ))}
      </div>
    </div>
  );
}
