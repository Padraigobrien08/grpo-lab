import { RunsTable } from "./runs-table";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-8">
      <header className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">Runs</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          GRPO post-training of Qwen2.5-0.5B-Instruct on GSM8K. Final values are each run&apos;s
          last logged step; eval accuracy is greedy, on a fixed 500-problem test subset. Open a
          run for its curves, then step into the rollout groups behind them.
        </p>
      </header>
      <RunsTable />
    </main>
  );
}
