/**
 * "What we did": the project's lab notebook, rendered at /journal, newest first.
 *
 * To add an entry, append to JOURNAL (oldest first here, so the file reads as a history).
 * House style, after nanogpt-pob.dev: the title states the finding, not the task; every figure
 * is measured (say where it came from in the body); the caveat says what went wrong or what the
 * number does not show. Keep paragraphs short. Inline `code` in backticks is rendered as code.
 */

export type JournalEntry = {
  id: string;
  date: string; // ISO date the work landed
  milestone: string; // e.g. "M3 · Evaluation"
  status?: "in-progress";
  title: string;
  body: string[];
  figures?: { value: string; label: string }[];
  caveat?: string;
  links?: { label: string; href: string }[];
};

const PR = (n: number) => ({
  label: `PR #${n}`,
  href: `https://github.com/Padraigobrien08/grpo-lab/pull/${n}`,
});

/** The strip under the page title: the project's current state in a few measured facts. */
export const STATUS = [
  "85 tests green",
  "Qwen2.5-0.5B on GSM8K",
  "+6.0 pts greedy over base",
  "M7 done: $5.87 of GPU",
];

export const JOURNAL: JournalEntry[] = [
  {
    id: "scaffold",
    date: "2026-11-22",
    milestone: "M0–M1 · Algorithm core",
    title: "A GRPO loop small enough to check by hand",
    body: [
      "We wrote the parts this project exists to show from scratch: group sampling, group-relative advantages, the clipped policy loss with its aggregation rules for GRPO, Dr. GRPO and DAPO, and the k3 KL estimator against a frozen reference. Transformers supplies the model and nothing else.",
      "The tests pin down what's easy to get subtly wrong: on-policy, the loss gradient equals the negative advantage exactly; summed micro-batch losses equal the full-batch loss; and the log-probs the loss scores come from the distribution `generate()` actually sampled, temperature included. On a toy task, a tiny random model goes from chance to emitting the target token almost every time.",
    ],
    figures: [
      { value: "0.06 → 0.8+", label: "toy-task reward, chance to learned, in 30 steps" },
    ],
    caveat:
      "Without LR warmup, Adam's first full-size step collapsed Qwen onto the format reward in one update, so the smoke config got warmup before anything real ran.",
  },
  {
    id: "prompt-starved",
    date: "2026-11-22",
    milestone: "M2 · Before the first GPU",
    title: "The prompt format was starving the gradient",
    body: [
      "Probing the base model before renting a GPU paid for itself. With GSM8K's own `#### <n>` answer format, Qwen2.5-0.5B-Instruct scored about 2% greedy. With Qwen's `\\boxed{}` math prompt it scored about 44%. GRPO learns only from groups whose samples disagree: at 6% sample accuracy, roughly 60% of groups of eight would be all-wrong, with zero advantage and no gradient.",
      "The same probe turned up a second silent failure: at a learning rate of 1e-6, AdamW updates to bf16 weights round to zero, so the model would have trained for hours without moving. Weights now stay fp32 with compute under bf16 autocast, and the config refuses bf16 weights outright.",
    ],
    figures: [
      { value: "~2% → ~44%", label: "base greedy accuracy, `####` prompt vs `\\boxed{}`" },
      { value: "~60%", label: "of groups that would carry no signal at 6% accuracy" },
    ],
    links: [PR(1)],
  },
  {
    id: "first-run",
    date: "2026-11-24",
    milestone: "M2 · First real run",
    title: "First GSM8K run on a rented GPU: 49.6% to 54.4%",
    body: [
      "300 steps on one A40 in about two hours: 16 prompts a step, eight samples each, KL to the reference at β = 0.04. Greedy accuracy on a fixed 500-problem test slice, scored every 25 steps, rose from 49.6% to 54.4%.",
      "Most of the work was the box, not the model. RunPod hosts with drivers older than CUDA 13 can't run the locked torch build, so setup now fails fast. Piped into `tee`, Python block-buffers stdout, and the log sat frozen for minutes at a time. The pod's own ID lives in a file that non-interactive shells skip, which meant the self-stop at the end ran with an empty ID until we sourced it explicitly.",
    ],
    figures: [{ value: "+4.8 pts", label: "one seed, 500-problem slice, step 0 to 300" }],
    caveat:
      "One seed on a 500-problem slice carries a standard error of about 2.2 points: a sign of life, and M3's job was to turn it into a result.",
    links: [PR(1), { label: "The run", href: "/run?id=gsm8k-grpo-20261124-122830-3d7458" }],
  },
  {
    id: "three-seeds",
    date: "2026-11-24",
    milestone: "M3 · Evaluation",
    title: "+6.0 points, and the gain shrinks as you sample more",
    body: [
      "Three seeds, each scored on all 1,319 GSM8K test problems at its final checkpoint, against the base model on the same problems. Greedy accuracy went from 47.6% to 53.6%. The base and trained intervals don't overlap, and each seed beats base on its own (paired McNemar p ≤ 1.1e-05).",
      "pass@k tells the more interesting story. The gain peaks at pass@1 (+15.3 points) and falls to +6.3 by pass@16, so most of what GRPO does here is make the model reliably produce answers it could already sample; it rarely teaches it to solve a problem it never could.",
    ],
    figures: [
      { value: "47.6% → 53.6%", label: "greedy, full test set, mean of 3 seeds" },
      { value: "+6.0", label: "points, 95% CI [+4.7, +7.3] across seeds" },
      { value: "+15.3 → +6.3", label: "gain at pass@1 vs pass@16" },
    ],
    caveat:
      "The in-training eval curve runs on test problems, and picking its best-looking step would inflate the result. So the headline always scores the final checkpoint.",
    links: [PR(2), { label: "Compare", href: "/compare" }],
  },
  {
    id: "dashboard",
    date: "2026-11-25",
    milestone: "M4 · Dashboard",
    title: "The run store is the only contract, so the dashboard needs no GPU",
    body: [
      "Trainer and dashboard never talk. The trainer writes SQLite metrics and Parquet rollouts, a read-only API serves them, and `export-static` snapshots every response to JSON for a static site that can read either source. That separation is why we could build and host the dashboard with the GPU switched off.",
      "Three views: Runs (final metrics and the config keys that differ), Run detail (curves, click through to the samples behind them) and the Rollout inspector (all eight samples of a prompt with their rewards and advantages, math rendered).",
    ],
    caveat:
      "GSM8K writes prices as `$3`, so a lone `$` can't mean math; the inspector renders `\\(…\\)`, `\\[…\\]` and `$$…$$` and leaves single dollars alone.",
    links: [PR(4), PR(5)],
  },
  {
    id: "mvp",
    date: "2026-11-26",
    milestone: "MVP · Close-out",
    title: "The dashboard went live, and one merge would have swapped it for its own API",
    body: [
      "Closing the MVP meant a clean-clone reproduction (locked install, tests, the toy run and a GSM8K smoke run, each recording its commit), the result figure, and deploying the dashboard: a prebuilt upload to Vercel, because the snapshot comes from a local run store that isn't in git.",
      "Linking the Vercel project quietly connected it to the GitHub repo, and its git build auto-detected the FastAPI app and deployed that, with no data, as the close-out PR's preview. Merging would have done the same to production. A root `vercel.json` now turns git deploys off.",
    ],
    caveat:
      "Checking that protected preview with `vercel curl` silently added a deployment-protection bypass secret to the project; we revoked it the same evening.",
    links: [PR(6), { label: "Live dashboard", href: "https://grpo-lab-seven.vercel.app" }],
  },
  {
    id: "token-traces",
    date: "2026-11-26",
    milestone: "Token traces",
    title: "Per-token traces without a single extra forward pass",
    body: [
      "Whenever it logs rollouts, the trainer now also keeps per-token log-prob, entropy, KL and ratio for the first two groups, copied from tensors the update already computes. In the inspector you can colour each completion by any of them, on one scale per group, and compare the eight siblings token by token.",
      "The check that matters: on the toy task, traced entropy and KL averaged over every token equal the step's logged metrics, which rules out an off-by-one or a bad truncation. On a real GSM8K run, joining the per-token strings back up reproduced every traced completion exactly.",
    ],
    figures: [
      { value: "0", label: "extra forward passes" },
      { value: "~7 MB", label: "snapshot JSON per 300-step run, projected from ~50 bytes a token" },
    ],
    caveat:
      "With one optimisation pass per batch the ratio is exactly 1, true of every run so far, so that view stays disabled until a run uses μ > 1. And the deployed runs predate traces entirely, so they have no heatmap.",
    links: [PR(7)],
  },
  {
    id: "compare-ux",
    date: "2026-11-26",
    milestone: "Dashboard · Compare and usability",
    title: "Seeds pooled, and each config named by what it changes",
    body: [
      "The Compare page groups runs by training config, pools their seeds into a mean line with a 95% t-band, and labels every group by its difference from the baseline, so an ablation reads as `algo.loss_aggregation=constant` rather than a run name.",
      "A usability pass followed. On a phone the rollout-step ticks were 12 px wide and 10 px apart, so they became a step picker; the step slider announced its index (“15”) instead of “Step 150 of 300”; and a bad link left you on a bare 404 with no way back.",
    ],
    caveat:
      "Grouping ignores logging settings. The new runs record `log.trace_groups`, which the M3 runs predate, and without that rule a logging flag would have shown up as an ablation.",
    links: [{ label: "Compare", href: "/compare" }],
  },
  {
    id: "m7",
    date: "2026-11-27",
    milestone: "M7 · RQ1 ablations",
    title: "Neither recipe moves accuracy at 0.5B, but DAPO-lite halves the entropy getting there",
    body: [
      "Which GRPO design choices matter at this scale? Dr. GRPO drops the per-group std scaling and uses a constant loss normaliser, while DAPO-lite switches to a token-level mean, clip-higher at 0.28 and no KL. We ran seeds 0 and 1 of each on four pods at once and scored every final checkpoint on all 1,319 test problems, against M3's three vanilla seeds as the baseline.",
      "Accuracy barely moved. Dr. GRPO scored 53.0% and 52.5% (+5.1 points over base), DAPO-lite 55.3% and 54.4% (+7.2), vanilla 53.6% (+6.0). Against vanilla, Dr. GRPO comes out −0.9 points (Welch 95% CI [−2.6, +0.9]) and DAPO-lite +1.2 ([−5.8, +8.2]); with two seeds each, neither gap is distinguishable from zero.",
      "The policies, though, ended up in different places. Over the last 20 training steps DAPO-lite's entropy averaged 0.100 against vanilla's 0.203, its answers ran 249 tokens instead of 291, and in 46% of its groups all eight samples agreed (31% for vanilla), so nearly half its batches taught it nothing. Dr. GRPO matched vanilla on all three and drifted half as far from the reference model (KL 0.006 against 0.012).",
    ],
    figures: [
      { value: "−0.9 / +1.2", label: "points vs vanilla, Dr. GRPO / DAPO-lite; both CIs cross zero" },
      { value: "0.100 vs 0.203", label: "policy entropy, DAPO-lite vs vanilla, last 20 steps" },
      { value: "$5.87", label: "of GPU for all four runs, read off the RunPod balance" },
    ],
    caveat:
      "Two seeds per variant is thin, and the Welch intervals round their degrees of freedom down, so they err wide. DAPO-lite collapsing its entropy without losing accuracy held for 300 steps; it may not hold for longer runs, and with 46% of groups carrying no signal, DAPO's dynamic sampling (which drops those groups and resamples) is exactly the piece this run left out. A third seed of each, and that sampler, come next.",
    links: [
      { label: "Compare", href: "/compare" },
      { label: "PR #8", href: "https://github.com/Padraigobrien08/grpo-lab/pull/8" },
    ],
  },
];
