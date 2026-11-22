import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

// Placeholder until M4 (Runs list, Run detail, Rollout inspector).
const views = [
  { name: "Runs", detail: "Every run with config diff, status and final metrics." },
  { name: "Run detail", detail: "Reward, KL, entropy, clip fraction and length curves." },
  { name: "Rollout inspector", detail: "All G samples per prompt with rewards and advantages." },
];

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-4 py-16">
      <header className="flex flex-col gap-3">
        <Badge variant="secondary" className="w-fit">
          M0 scaffold
        </Badge>
        <h1 className="text-3xl font-semibold tracking-tight">grpo-lab</h1>
        <p className="text-muted-foreground">
          From-scratch GRPO post-training for small language models, and a dashboard to see
          inside training.
        </p>
      </header>
      <section className="grid gap-4 sm:grid-cols-3">
        {views.map((v) => (
          <Card key={v.name} size="sm">
            <CardHeader>
              <CardTitle>{v.name}</CardTitle>
              <CardDescription>{v.detail}</CardDescription>
            </CardHeader>
            <CardContent>
              <Badge variant="outline">M4</Badge>
            </CardContent>
          </Card>
        ))}
      </section>
    </main>
  );
}
