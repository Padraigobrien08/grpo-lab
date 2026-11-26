"use client";

import type { TraceRow } from "@/lib/api";
import {
  formatValue,
  type Scale,
  TOKEN_VIEWS,
  type TokenView,
  tokenTitle,
  viewValues,
} from "@/lib/traces";
import { cn } from "@/lib/utils";

// Strongest tint, as a share of the hue mixed into the background. Kept below ~60% so token
// text (always foreground ink, never the series colour) stays readable at full strength.
const MAX_TINT = 55;

function tint(strength: number, above: boolean): string | undefined {
  if (strength <= 0) return undefined;
  const hue = above ? "var(--chart-1)" : "var(--chart-2)";
  return `color-mix(in oklch, ${hue} ${Math.round(strength * MAX_TINT)}%, transparent)`;
}

/** Segmented control for the group's token view; views with no data are disabled with a reason. */
export function TokenViewPicker({
  view,
  onChange,
  unavailable,
}: {
  view: TokenView;
  onChange: (v: TokenView) => void;
  unavailable: Partial<Record<TokenView, string>>;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Colour tokens by"
      className="inline-flex w-fit flex-wrap rounded-lg border bg-muted/40 p-0.5 text-xs"
    >
      {TOKEN_VIEWS.map((t) => {
        const reason = unavailable[t.id];
        const active = view === t.id;
        return (
          <button
            key={t.id}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={!!reason}
            title={reason ?? t.hint}
            onClick={() => onChange(t.id)}
            className={cn(
              "rounded-md px-2.5 py-1 font-medium text-muted-foreground transition-colors",
              "hover:text-foreground disabled:pointer-events-none disabled:opacity-40",
              active && "bg-background text-foreground shadow-xs",
            )}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

/** Colour key for the group's shared scale. */
export function ScaleLegend({ view, scale }: { view: TokenView; scale: Scale }) {
  const hint = TOKEN_VIEWS.find((t) => t.id === view)?.hint;
  const stops = scale.diverging
    ? `linear-gradient(to right, ${tint(1, false)}, transparent, ${tint(1, true)})`
    : `linear-gradient(to right, transparent, ${tint(1, true)})`;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5 font-mono tabular-nums">
        <span>{scale.diverging ? `≤ ${formatValue(1 / scale.max)}` : "0"}</span>
        <span className="h-2.5 w-24 rounded-sm border" style={{ background: stops }} aria-hidden />
        <span>≥ {formatValue(scale.max)}</span>
      </span>
      <span>{hint}. Scale shared by this group; hover a token for all its values.</span>
    </div>
  );
}

/** The completion as coloured tokens. Whitespace and newlines are kept as generated. */
export function TokenHeatmap({
  row,
  view,
  scale,
}: {
  row: TraceRow;
  view: Exclude<TokenView, "text">;
  scale: Scale;
}) {
  const values = viewValues(row, view);
  return (
    <div className="max-h-96 overflow-y-auto px-3 py-3 font-mono text-[0.8rem] leading-6 break-words whitespace-pre-wrap">
      {row.tokens.map((tok, i) => {
        const special = /^<\|.*\|>$/.test(tok);
        const { strength, above } = values ? scale.at(values[i]) : { strength: 0, above: true };
        return (
          <span
            key={i}
            title={tokenTitle(row, i)}
            className={cn("rounded-[3px]", special && "px-0.5 text-muted-foreground")}
            style={{ background: tint(strength, above) }}
          >
            {/* a whitespace-only token with a line break would be an invisible box: mark it */}
            {tok.includes("\n") && tok.trim() === "" ? (
              <>
                <span className="text-muted-foreground/60" aria-hidden>
                  ↵
                </span>
                {tok}
              </>
            ) : (
              tok
            )}
          </span>
        );
      })}
    </div>
  );
}
