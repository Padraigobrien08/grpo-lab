"use client";

import "katex/dist/katex.min.css";

import { useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { RolloutRow } from "@/lib/api";
import { renderTex, splitMath } from "@/lib/math";
import { components } from "@/lib/rollouts";
import { cn } from "@/lib/utils";

export function SampleCard({ row, maxAbsAdvantage }: { row: RolloutRow; maxAbsAdvantage: number }) {
  const comps = components(row);
  const correct = comps.correct === 1;
  const extras = Object.entries(comps).filter(([k]) => k !== "correct");
  return (
    <article className="flex flex-col rounded-lg border bg-card" aria-label={`Sample ${row.sample_idx}`}>
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b px-3 py-2 text-xs tabular-nums">
        <span className="flex items-center gap-2">
          <span className="font-mono text-muted-foreground">#{row.sample_idx}</span>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 font-medium",
              correct ? "bg-correct-soft text-correct" : "bg-incorrect-soft text-incorrect",
            )}
          >
            <span className={cn("size-1.5 rounded-full", correct ? "bg-correct" : "bg-incorrect")} />
            {correct ? "Correct" : "Incorrect"}
          </span>
        </span>
        <dl className="flex items-center gap-4">
          <Stat label="reward" value={row.reward_total.toFixed(2)} />
          {extras.map(([k, v]) => (
            <Stat key={k} label={k} value={String(v)} />
          ))}
          <Stat label="tokens" value={String(row.length)} />
        </dl>
        {row.truncated && <Badge variant="destructive">truncated</Badge>}
        <AdvantageBar value={row.advantage} maxAbs={maxAbsAdvantage} />
      </header>
      <Completion text={row.completion} />
    </article>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1">
      <dt className="text-muted-foreground">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

/** Signed bar from zero: how far the group-normalised advantage pushes this sample. */
function AdvantageBar({ value, maxAbs }: { value: number; maxAbs: number }) {
  const frac = maxAbs > 0 ? Math.min(1, Math.abs(value) / maxAbs) : 0;
  return (
    <span className="ml-auto flex items-center gap-2" title="Group-normalised advantage">
      <span className="text-muted-foreground">adv</span>
      <span className="relative h-2 w-24 rounded-full bg-muted" aria-hidden>
        <span className="absolute inset-y-0 left-1/2 w-px bg-foreground/30" />
        <span
          className={cn(
            "absolute inset-y-0 rounded-full",
            value > 0 ? "left-1/2 bg-correct" : "right-1/2 bg-incorrect",
          )}
          style={{ width: `${frac * 50}%` }}
        />
      </span>
      <span
        className={cn(
          "w-11 text-right font-mono",
          value > 0 && "text-correct",
          value < 0 && "text-incorrect",
          value === 0 && "text-muted-foreground",
        )}
      >
        {value > 0 ? "+" : value < 0 ? "−" : ""}
        {Math.abs(value).toFixed(2)}
      </span>
    </span>
  );
}

/** The completion with math rendered; long answers are clamped with a "show all" toggle. */
function Completion({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const check = () => setOverflows(el.scrollHeight > el.clientHeight + 1);
    check();
    const ro = new ResizeObserver(check);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <div className="flex flex-col">
      <div
        ref={ref}
        className={cn(
          "px-3 py-3 text-sm leading-relaxed break-words whitespace-pre-wrap",
          !open && "max-h-72 overflow-hidden",
          !open && overflows && "[mask-image:linear-gradient(to_bottom,black_70%,transparent)]",
        )}
      >
        {splitMath(text).map((seg, i, all) => {
          if (seg.kind === "text") {
            // display math is a block already: drop the newline either side of it
            const isBlock = (s: (typeof all)[number] | undefined) => s?.kind === "math" && s.display;
            let t = seg.text;
            if (isBlock(all[i - 1])) t = t.replace(/^[ \t]*\n/, "");
            if (isBlock(all[i + 1])) t = t.replace(/\n[ \t]*$/, "");
            return <span key={i}>{t}</span>;
          }
          const html = renderTex(seg.tex, seg.display);
          const answer = seg.tex.includes("\\boxed");
          if (html === null) {
            return (
              <code key={i} className="font-mono text-xs">
                {seg.source}
              </code>
            );
          }
          const Tag = seg.display ? "div" : "span";
          return (
            <Tag
              key={i}
              className={cn(
                "whitespace-normal",
                seg.display && "my-1 overflow-x-auto",
                answer && "rounded bg-answer px-1",
                answer && seg.display && "w-fit",
              )}
              // KaTeX escapes the TeX it renders, and trust is off (see lib/math.ts)
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
        })}
      </div>
      {(overflows || open) && (
        <Button
          variant="ghost"
          size="sm"
          className="m-1 self-start text-muted-foreground"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
        >
          {open ? "Show less" : "Show full answer"}
        </Button>
      )}
    </div>
  );
}
