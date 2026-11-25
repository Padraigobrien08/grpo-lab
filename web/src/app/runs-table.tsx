"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { ErrorBlock, LoadingBlock } from "@/components/states";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { paths, type RunSummary, useJson } from "@/lib/api";
import { configOutliers } from "@/lib/config-diff";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
// One precision per column, so values line up and compare at a glance.
const FINAL_METRICS: { key: string; label: string; format: (v: number) => string }[] = [
  { key: "eval/accuracy", label: "Eval acc.", format: pct },
  { key: "reward/mean", label: "Reward", format: (v) => v.toFixed(3) },
  { key: "kl", label: "KL", format: (v) => v.toFixed(4) },
  { key: "entropy", label: "Entropy", format: (v) => v.toFixed(3) },
];

const runHref = (id: string) => `/run?id=${encodeURIComponent(id)}`;

export function RunsTable() {
  const runs = useJson<RunSummary[]>(paths.runs());
  const router = useRouter();
  if (runs.state === "loading") return <LoadingBlock rows={4} />;
  if (runs.state === "error") return <ErrorBlock error={runs.error} />;
  if (runs.data.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No runs in this store yet. Train one with <code>uv run train</code>, then re-export.
      </p>
    );
  }

  // name and seed have their own columns, so they never count as a config difference
  const outliers = configOutliers(
    runs.data.map((r) => r.config),
    ["name", "seed"],
  );

  return (
    <div className="overflow-x-auto rounded-lg border">
      <Table className="tabular-nums">
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="sticky left-0 z-10 bg-background">Run</TableHead>
            <TableHead className="text-right">Seed</TableHead>
            <TableHead className="text-right">Steps</TableHead>
            {FINAL_METRICS.map((m) => (
              <TableHead key={m.key} className="text-right">
                {m.label}
              </TableHead>
            ))}
            <TableHead>Differs from the others</TableHead>
            <TableHead>Commit</TableHead>
            <TableHead className="text-right">Started</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.data.map((run, i) => {
            const started = new Date(run.started_at * 1000).toISOString();
            return (
              <TableRow
                key={run.id}
                className="group/row cursor-pointer"
                onClick={(e) => {
                  // the whole row opens the run; real links inside it keep their own target
                  if (!(e.target as HTMLElement).closest("a")) router.push(runHref(run.id));
                }}
              >
                <TableCell className="sticky left-0 z-10 bg-background group-hover/row:bg-[color-mix(in_oklab,var(--muted)_50%,var(--background))]">
                  <span className="flex items-center gap-2">
                    <Link
                      href={runHref(run.id)}
                      title={run.id}
                      className="font-medium hover:underline"
                    >
                      {run.name}
                    </Link>
                    {run.status !== "finished" && <Badge variant="outline">{run.status}</Badge>}
                  </span>
                </TableCell>
                <TableCell className="text-right">{run.seed ?? "—"}</TableCell>
                <TableCell className="text-right">{run.last_step ?? "—"}</TableCell>
                {FINAL_METRICS.map((m) => {
                  const v = run.last_metrics[m.key];
                  return (
                    <TableCell key={m.key} className="text-right">
                      {v === undefined ? "—" : m.format(v)}
                    </TableCell>
                  );
                })}
                <TableCell className="font-mono text-xs">
                  {outliers[i].length === 0 ? (
                    <span className="text-muted-foreground">—</span>
                  ) : (
                    outliers[i].map(([k, v]) => (
                      <div key={k}>
                        <span className="text-muted-foreground">{k}=</span>
                        <span className="font-medium">{v}</span>
                      </div>
                    ))
                  )}
                </TableCell>
                <TableCell className="font-mono text-xs">
                  {run.git_sha ? (
                    <a
                      href={`https://github.com/Padraigobrien08/grpo-lab/commit/${run.git_sha}`}
                      className="text-muted-foreground hover:text-foreground hover:underline"
                    >
                      {run.git_sha.slice(0, 7)}
                    </a>
                  ) : (
                    "—"
                  )}
                </TableCell>
                <TableCell
                  className="text-right whitespace-nowrap text-muted-foreground"
                  title={`${started.slice(0, 16).replace("T", " ")} UTC`}
                >
                  {started.slice(0, 10)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
