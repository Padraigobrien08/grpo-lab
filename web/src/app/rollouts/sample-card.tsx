import { Badge } from "@/components/ui/badge";
import type { RolloutRow } from "@/lib/api";
import { cn } from "@/lib/utils";
import { components, splitBoxed } from "@/lib/rollouts";

export function SampleCard({ row }: { row: RolloutRow }) {
  const comps = components(row);
  const correct = comps.correct === 1;
  const adv = row.advantage;
  return (
    <article
      className={cn(
        "flex flex-col gap-2 rounded-lg border p-3",
        correct ? "border-l-4 border-l-emerald-500" : "border-l-4 border-l-rose-400",
      )}
    >
      <header className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="font-mono text-muted-foreground">#{row.sample_idx}</span>
        <Badge variant={correct ? "secondary" : "outline"}>
          {correct ? "correct" : "incorrect"}
        </Badge>
        <span className="tabular-nums">reward {row.reward_total.toFixed(2)}</span>
        <span
          className={cn(
            "font-mono tabular-nums",
            adv > 0 && "text-emerald-600 dark:text-emerald-400",
            adv < 0 && "text-rose-600 dark:text-rose-400",
            adv === 0 && "text-muted-foreground",
          )}
          title="Group-normalised advantage"
        >
          adv {adv > 0 ? "+" : ""}
          {adv.toFixed(2)}
        </span>
        <span className="text-muted-foreground tabular-nums">{row.length} tok</span>
        {row.truncated && <Badge variant="destructive">truncated</Badge>}
        {Object.entries(comps)
          .filter(([k]) => k !== "correct")
          .map(([k, v]) => (
            <span key={k} className="text-muted-foreground">
              {k} {v}
            </span>
          ))}
      </header>
      <pre className="max-h-80 overflow-y-auto font-sans text-sm leading-relaxed whitespace-pre-wrap">
        {splitBoxed(row.completion).map((part, i) =>
          part.boxed ? (
            <mark key={i} className="rounded bg-amber-200/70 px-0.5 dark:bg-amber-500/30">
              {part.text}
            </mark>
          ) : (
            <span key={i}>{part.text}</span>
          ),
        )}
      </pre>
    </article>
  );
}
