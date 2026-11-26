import type { RolloutRow } from "@/lib/api";
import { cn } from "@/lib/utils";
import { components } from "@/lib/rollouts";

/** One square per sample in the group: solid = correct / incorrect, hollow = truncated. */
export function Pips({ samples, className }: { samples: RolloutRow[]; className?: string }) {
  const correct = samples.filter((s) => components(s).correct === 1).length;
  return (
    <span
      className={cn("inline-flex gap-0.5", className)}
      role="img"
      aria-label={`${correct} of ${samples.length} correct`}
    >
      {samples.map((s) => {
        const ok = components(s).correct === 1;
        return (
          <span
            key={s.sample_idx}
            className={cn(
              "size-2.5 rounded-[2px]",
              ok ? "bg-correct" : s.truncated ? "border border-incorrect" : "bg-incorrect",
            )}
          />
        );
      })}
    </span>
  );
}
