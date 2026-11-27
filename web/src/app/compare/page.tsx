import { Suspense } from "react";

import { LoadingBlock } from "@/components/states";

import { CompareView } from "./compare-view";

export default function ComparePage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-8">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">Compare</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Runs grouped by training config, with seeds of the same config pooled. Name, seed,
          device and logging settings don&apos;t split a group. The group with the most runs is the
          baseline; every other group is labelled by what it changes. Headline numbers belong to
          the full-test-set evals in the README, not these in-training curves.
        </p>
      </header>
      <Suspense fallback={<LoadingBlock rows={6} />}>
        <CompareView />
      </Suspense>
    </main>
  );
}
