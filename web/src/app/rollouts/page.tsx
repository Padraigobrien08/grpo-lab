import { Suspense } from "react";

import { InspectorSkeleton } from "@/components/states";

import { RolloutInspector } from "./inspector";

// /rollouts?id=<run>&step=<step>&prompt=<prompt_id>, all query params (see app/run/page.tsx).
export default function RolloutsPage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-8">
      <Suspense fallback={<InspectorSkeleton />}>
        <RolloutInspector />
      </Suspense>
    </main>
  );
}
