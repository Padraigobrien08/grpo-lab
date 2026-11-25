"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { ErrorBlock, LoadingBlock } from "@/components/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { paths, type RolloutRow, type RunDetail, useJson } from "@/lib/api";
import { cn } from "@/lib/utils";
import { components, type Group, groupRollouts } from "@/lib/rollouts";

import { SampleCard } from "./sample-card";

export function RolloutInspector() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const id = params.get("id");
  const run = useJson<RunDetail>(id && paths.run(id));
  const steps = useJson<number[]>(id && paths.rolloutSteps(id));

  const stepList = steps.state === "ready" ? steps.data : [];
  const requested = Number(params.get("step"));
  const step = stepList.includes(requested) ? requested : (stepList[0] ?? null);
  const rows = useJson<RolloutRow[]>(id && step !== null ? paths.rollouts(id, step) : null);

  if (!id) return <ErrorBlock error="No run id in the URL (expected /rollouts?id=...)" />;
  for (const r of [run, steps, rows]) if (r.state === "error") return <ErrorBlock error={r.error} />;
  if (run.state !== "ready" || steps.state !== "ready") return <LoadingBlock rows={8} />;
  if (stepList.length === 0 || step === null) {
    return <p className="text-sm text-muted-foreground">This run logged no rollouts.</p>;
  }

  const go = (next: { step?: number; prompt?: string | null }) => {
    const q = new URLSearchParams(params.toString());
    if (next.step !== undefined) {
      q.set("step", String(next.step));
      q.delete("prompt"); // prompts differ between steps
    }
    if (next.prompt !== undefined) {
      if (next.prompt) q.set("prompt", next.prompt);
      else q.delete("prompt");
    }
    router.replace(`${pathname}?${q.toString()}`, { scroll: false });
  };

  const stepIdx = stepList.indexOf(step);
  const groups = rows.state === "ready" ? groupRollouts(rows.data) : [];
  const selected = groups.find((g) => g.promptId === params.get("prompt")) ?? groups[0];

  return (
    <>
      <header className="flex flex-col gap-3">
        <Link
          href={`/run?id=${encodeURIComponent(id)}`}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          ← {run.data.name}
        </Link>
        <div className="flex flex-wrap items-baseline gap-3">
          <h1 className="text-2xl font-semibold tracking-tight">Rollout inspector</h1>
          <span className="font-mono text-xs text-muted-foreground">{run.data.id}</span>
        </div>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Each training step samples {groups[0]?.samples.length ?? "G"} completions per prompt.
          Advantages are normalised within the group, so a sample is pushed up only relative to
          its siblings, and a group where every sample scored the same teaches nothing.
        </p>
      </header>

      <section className="flex items-center gap-3 rounded-lg border p-4 sm:gap-4">
        <span className="hidden text-sm text-muted-foreground sm:inline">Step</span>
        <Button
          variant="outline"
          size="sm"
          disabled={stepIdx <= 0}
          onClick={() => go({ step: stepList[stepIdx - 1] })}
          aria-label="Previous logged step"
        >
          ←
        </Button>
        <div className="min-w-0 flex-1">
          <Slider
            value={[stepIdx]}
            min={0}
            max={stepList.length - 1}
            step={1}
            onValueChange={(v) => go({ step: stepList[Array.isArray(v) ? v[0] : v] })}
            aria-label="Logged step"
          />
        </div>
        <Button
          variant="outline"
          size="sm"
          disabled={stepIdx >= stepList.length - 1}
          onClick={() => go({ step: stepList[stepIdx + 1] })}
          aria-label="Next logged step"
        >
          →
        </Button>
        <span className="shrink-0 font-mono text-sm tabular-nums">
          {step} <span className="text-muted-foreground">/ {stepList.at(-1)}</span>
        </span>
      </section>

      {rows.state !== "ready" ? (
        <LoadingBlock rows={8} />
      ) : (
        <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
          <GroupList groups={groups} selected={selected} onSelect={(p) => go({ prompt: p })} />
          {selected && <GroupView group={selected} />}
        </div>
      )}
    </>
  );
}

function GroupList({
  groups,
  selected,
  onSelect,
}: {
  groups: Group[];
  selected: Group | undefined;
  onSelect: (promptId: string) => void;
}) {
  return (
    <nav aria-label="Prompt groups" className="flex flex-col gap-1">
      <p className="px-1 pb-1 text-xs text-muted-foreground">
        {groups.length} prompts · {groups.filter((g) => g.zeroAdvantage).length} with zero
        advantage
      </p>
      <div className="flex max-h-[70vh] flex-col gap-1 overflow-y-auto pr-1">
        {groups.map((g) => (
          <button
            key={g.promptId}
            type="button"
            onClick={() => onSelect(g.promptId)}
            aria-current={g === selected}
            className={cn(
              "flex flex-col gap-1 rounded-lg border px-3 py-2 text-left text-xs transition-colors hover:bg-muted",
              g === selected && "border-foreground/40 bg-muted",
            )}
          >
            <span className="flex items-center justify-between gap-2">
              <span className="font-mono text-muted-foreground">{g.promptId}</span>
              <span className="tabular-nums">
                {g.correct}/{g.samples.length} correct
              </span>
            </span>
            <span className="line-clamp-2">{g.prompt}</span>
            {g.zeroAdvantage && (
              <span className="text-muted-foreground">all rewards equal · no signal</span>
            )}
          </button>
        ))}
      </div>
    </nav>
  );
}

function GroupView({ group }: { group: Group }) {
  const names = [...new Set(group.samples.flatMap((s) => Object.keys(components(s))))];
  return (
    <section className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 rounded-lg border p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs text-muted-foreground">{group.promptId}</span>
          <Badge variant="outline">
            {group.correct}/{group.samples.length} correct
          </Badge>
          <Badge variant="outline">
            reward {group.rewardMean.toFixed(2)} ± {group.rewardStd.toFixed(2)}
          </Badge>
          {group.zeroAdvantage && <Badge variant="secondary">zero advantage</Badge>}
        </div>
        <p className="text-sm whitespace-pre-wrap">{group.prompt}</p>
        <p className="text-xs text-muted-foreground">
          Reward = {names.join(" + ") || "total"} (weighted). Per-token traces (KL, entropy,
          ratio) were not recorded for these runs, so there is no token heatmap yet.
        </p>
      </div>
      <div className="grid gap-3 xl:grid-cols-2">
        {group.samples.map((s) => (
          <SampleCard key={s.sample_idx} row={s} />
        ))}
      </div>
    </section>
  );
}
