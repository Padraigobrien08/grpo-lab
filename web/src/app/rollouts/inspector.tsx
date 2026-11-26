"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef } from "react";

import { BackLink } from "@/components/back-link";
import { ErrorBlock, InspectorSkeleton } from "@/components/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { paths, type RolloutRow, type RunDetail, useJson } from "@/lib/api";
import { components, type Group, groupRollouts } from "@/lib/rollouts";
import { cn } from "@/lib/utils";

import { Pips } from "./pips";
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
  if (run.state !== "ready" || steps.state !== "ready") return <InspectorSkeleton />;
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
      <header className="flex flex-col gap-4">
        <BackLink href={`/run?id=${encodeURIComponent(id)}`}>{run.data.name}</BackLink>
        <div className="flex flex-col gap-1.5">
          <h1 className="text-2xl font-semibold tracking-tight">Rollouts</h1>
          <p className="max-w-3xl text-sm text-muted-foreground">
            Each step samples {groups[0]?.samples.length ?? "G"} completions per prompt. Advantages
            are normalised within the group: a sample is pushed up only relative to its siblings,
            and a group where every sample scored the same teaches nothing.
          </p>
        </div>
      </header>

      <section
        className="flex items-center gap-3 rounded-lg border p-4 sm:gap-4"
        aria-label="Logged step"
      >
        <span className="hidden text-sm text-muted-foreground sm:inline">Step</span>
        <Button
          variant="outline"
          size="icon-sm"
          disabled={stepIdx <= 0}
          onClick={() => go({ step: stepList[stepIdx - 1] })}
          aria-label="Previous logged step"
        >
          <ChevronLeft aria-hidden />
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
          size="icon-sm"
          disabled={stepIdx >= stepList.length - 1}
          onClick={() => go({ step: stepList[stepIdx + 1] })}
          aria-label="Next logged step"
        >
          <ChevronRight aria-hidden />
        </Button>
        <span className="shrink-0 font-mono text-sm tabular-nums">
          {step} <span className="text-muted-foreground">of {stepList.at(-1)}</span>
        </span>
      </section>

      {rows.state !== "ready" ? (
        <InspectorSkeleton bodyOnly />
      ) : (
        <div className="grid min-w-0 gap-4 lg:grid-cols-[17rem_minmax(0,1fr)]">
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
  const silent = groups.filter((g) => g.zeroAdvantage).length;
  const current = useRef<HTMLButtonElement>(null);
  // keep the selected group in view, e.g. when arriving from a shared ?prompt= link
  useEffect(() => {
    current.current?.scrollIntoView({ block: "nearest", inline: "start" });
  }, [selected]);
  return (
    <nav aria-label="Prompt groups" className="flex min-w-0 flex-col gap-2">
      <p className="text-xs text-muted-foreground">
        {groups.length} prompts · {silent} with no learning signal
      </p>
      {/* a horizontal strip on narrow screens, so the samples stay one scroll away */}
      <div className="-mx-4 flex snap-x gap-2 overflow-x-auto px-4 pb-1 lg:mx-0 lg:max-h-[calc(100vh-14rem)] lg:snap-none lg:flex-col lg:overflow-x-visible lg:overflow-y-auto lg:px-0 lg:pr-1">
        {groups.map((g) => (
          <button
            key={g.promptId}
            ref={g === selected ? current : undefined}
            type="button"
            onClick={() => onSelect(g.promptId)}
            aria-current={g === selected}
            className={cn(
              "flex w-60 shrink-0 snap-start flex-col gap-1.5 rounded-lg border px-3 py-2 text-left text-xs transition-colors hover:bg-muted lg:w-auto",
              g === selected && "border-foreground/30 bg-muted",
            )}
          >
            <span className="flex items-center justify-between gap-2">
              <Pips samples={g.samples} />
              <span className="tabular-nums text-muted-foreground">
                {g.correct}/{g.samples.length}
              </span>
            </span>
            <span className="line-clamp-2 text-foreground/90">{g.prompt}</span>
            <span className="flex items-center justify-between gap-2 text-muted-foreground">
              <span className="font-mono">{g.promptId}</span>
              {g.zeroAdvantage && <span>no signal</span>}
            </span>
          </button>
        ))}
      </div>
    </nav>
  );
}

function GroupView({ group }: { group: Group }) {
  const names = [...new Set(group.samples.flatMap((s) => Object.keys(components(s))))];
  const maxAbs = Math.max(...group.samples.map((s) => Math.abs(s.advantage)));
  return (
    <section className="flex min-w-0 flex-col gap-4" aria-label={`Group ${group.promptId}`}>
      <div className="flex flex-col gap-3 rounded-lg border p-4">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs tabular-nums">
          <span className="flex items-center gap-2">
            <Pips samples={group.samples} />
            <span>
              {group.correct} of {group.samples.length} correct
            </span>
          </span>
          <span className="text-muted-foreground">
            reward {group.rewardMean.toFixed(2)} ± {group.rewardStd.toFixed(2)}
          </span>
          {group.zeroAdvantage && <Badge variant="secondary">no learning signal</Badge>}
          <span className="ml-auto font-mono text-muted-foreground">{group.promptId}</span>
        </div>
        <p className="max-w-[75ch] text-base leading-relaxed whitespace-pre-wrap">{group.prompt}</p>
      </div>
      <div className="grid gap-3 xl:grid-cols-2">
        {group.samples.map((s) => (
          <SampleCard key={s.sample_idx} row={s} maxAbsAdvantage={maxAbs} />
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Reward = {names.join(" + ") || "total"} (weighted). Per-token traces (KL, entropy, ratio)
        weren&apos;t recorded for these runs, so there&apos;s no token heatmap yet.
      </p>
    </section>
  );
}
