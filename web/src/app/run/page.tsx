import { Suspense } from "react";

import { LoadingBlock } from "@/components/states";

import { RunDetailView } from "./run-detail";

// The run id is a query param (/run?id=...), not a dynamic segment, so this one static page
// serves every run: from the live API in development and from the snapshot when deployed.
export default function RunPage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-8">
      <Suspense fallback={<LoadingBlock rows={6} />}>
        <RunDetailView />
      </Suspense>
    </main>
  );
}
