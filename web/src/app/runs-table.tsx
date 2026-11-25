"use client";

import Link from "next/link";

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
import { differingKeys, flatten } from "@/lib/config-diff";
import { formatMetric } from "@/lib/metrics";

const FINAL_METRICS: [string, string][] = [
  ["eval/accuracy", "Eval acc."],
  ["reward/mean", "Reward"],
  ["kl", "KL"],
  ["entropy", "Entropy"],
];

export function RunsTable() {
  const runs = useJson<RunSummary[]>(paths.runs());
  if (runs.state === "loading") return <LoadingBlock rows={4} />;
  if (runs.state === "error") return <ErrorBlock error={runs.error} />;
  if (runs.data.length === 0) {
    return <p className="text-sm text-muted-foreground">No runs in this store yet.</p>;
  }

  // name and seed have their own columns, so leave them out of the diff
  const diffKeys = differingKeys(
    runs.data.map((r) => r.config),
    ["name", "seed"],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Run</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Seed</TableHead>
              <TableHead className="text-right">Steps</TableHead>
              {FINAL_METRICS.map(([key, label]) => (
                <TableHead key={key} className="text-right">
                  {label}
                </TableHead>
              ))}
              {diffKeys.length > 0 && <TableHead>Config diff</TableHead>}
              <TableHead>Commit</TableHead>
              <TableHead>Started</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.data.map((run) => {
              const flat = flatten(run.config);
              return (
                <TableRow key={run.id}>
                  <TableCell>
                    <Link
                      href={`/run?id=${encodeURIComponent(run.id)}`}
                      className="font-medium underline-offset-4 hover:underline"
                    >
                      {run.name}
                    </Link>
                    <div className="font-mono text-xs text-muted-foreground">{run.id}</div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={run.status === "finished" ? "secondary" : "outline"}>
                      {run.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{run.seed ?? "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">{run.last_step ?? "—"}</TableCell>
                  {FINAL_METRICS.map(([key]) => (
                    <TableCell key={key} className="text-right tabular-nums">
                      {formatMetric(run.last_metrics[key], key)}
                    </TableCell>
                  ))}
                  {diffKeys.length > 0 && (
                    <TableCell className="font-mono text-xs">
                      {diffKeys.map((k) => (
                        <div key={k}>
                          {k}={flat[k] ?? "—"}
                        </div>
                      ))}
                    </TableCell>
                  )}
                  <TableCell className="font-mono text-xs">
                    {run.git_sha ? (
                      <a
                        href={`https://github.com/Padraigobrien08/grpo-lab/commit/${run.git_sha}`}
                        className="underline-offset-4 hover:underline"
                      >
                        {run.git_sha.slice(0, 7)}
                      </a>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                  <TableCell className="text-xs whitespace-nowrap text-muted-foreground">
                    {new Date(run.started_at * 1000).toISOString().slice(0, 16).replace("T", " ")}{" "}
                    UTC
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
      {diffKeys.length === 0 && (
        <p className="text-xs text-muted-foreground">
          All runs share the same config apart from name and seed.
        </p>
      )}
    </div>
  );
}
