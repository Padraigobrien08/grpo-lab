import Link from "next/link";

import { Skeleton } from "@/components/ui/skeleton";

export function LoadingBlock({ rows = 3 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-2" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-8 w-full" />
      ))}
    </div>
  );
}

/** Run page placeholder: title, summary line, then the chart grid. */
export function RunSkeleton() {
  return (
    <div className="flex flex-col gap-6" aria-busy="true" aria-label="Loading">
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-16" />
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96 max-w-full" />
      </div>
      <Skeleton className="h-24 w-full" />
      <div className="grid gap-4 md:grid-cols-2">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-80" />
        ))}
      </div>
    </div>
  );
}

/** Inspector placeholder: prompt list beside the group's sample cards. */
export function InspectorSkeleton({ bodyOnly = false }: { bodyOnly?: boolean }) {
  return (
    <div className="flex flex-col gap-6" aria-busy="true" aria-label="Loading">
      {!bodyOnly && (
        <>
          <div className="flex flex-col gap-2">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-8 w-40" />
          </div>
          <Skeleton className="h-16 w-full" />
        </>
      )}
      <div className="grid gap-4 lg:grid-cols-[17rem_1fr]">
        <div className="hidden flex-col gap-2 lg:flex">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
        <div className="flex flex-col gap-3">
          <Skeleton className="h-28" />
          <div className="grid gap-3 xl:grid-cols-2">
            {Array.from({ length: 4 }, (_, i) => (
              <Skeleton key={i} className="h-64" />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function ErrorBlock({ error }: { error: string }) {
  const notFound = /\b404\b/.test(error);
  return (
    <div
      className="flex flex-col gap-2 rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm"
      role="alert"
    >
      <p className="font-medium">{notFound ? "Not found" : "Couldn\u2019t load data"}</p>
      <p className="text-muted-foreground">
        {notFound
          ? "This run or step isn\u2019t in the data. The link may be old, or the dashboard was exported before the run finished."
          : "The data didn\u2019t load. Reload the page; if it keeps failing, the snapshot or API may be missing this file."}
      </p>
      <Link href="/" className="w-fit text-sm font-medium underline underline-offset-4">
        Back to all runs
      </Link>
      <p className="font-mono text-xs text-muted-foreground/80">{error}</p>
    </div>
  );
}
