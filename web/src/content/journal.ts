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
  "M7 ablations running",
];

export const JOURNAL: JournalEntry[] = [
  {
    id: "scaffold",
    date: "2026-11-22",
    milestone: "M0–M1 · Algorithm core",
    title: "A GRPO loop small enough to check by hand",
    body: [
      "Everything that is the skill being shown is written from scratch: group sampling, group-relative advantages, the clipped policy loss with its three aggregation rules (GRPO, Dr. GRPO, DAPO), and the k3 KL estimator against a frozen reference. Transformers supplies the model and nothing else.",
      "The tests pin the parts that are easy to get subtly wrong. On-policy, the gradient of the loss is exactly the negative advantage. Summed micro-batch losses equal the full-batch loss. The log-probs the loss scores are the distribution `generate()` actually sampled from, temperature included. And on a toy task a tiny random model goes from chance to reliably emitting the target token.",
    ],
    figures: [
      { value: "0.06 → 0.8+", label: "toy-task reward, chance to learned, in 30 steps" },
    ],
    caveat:
      "Without LR warmup, Adam's first full-size step collapsed Qwen onto the format reward in a single update. The smoke config got warmup before anything real ran.",
  },
  {
    id: "prompt-starved",
    date: "2026-11-22",
    milestone: "M2 · Before the first GPU",
    title: "The prompt format was starving the gradient",
    body: [
      "Probing the base model before renting a GPU paid for itself. With GSM8K's own `#### <n>` answer format, Qwen2.5-0.5B-Instruct scored about 2% greedy. With Qwen's `\\boxed{}` math prompt it scored about 44%. GRPO learns only from groups whose samples disagree: at 6% sample accuracy, roughly 60% of groups of eight would be all-wrong, with zero advantage and no gradient.",
      "The same probe found a second silent failure. At a learning rate of 1e-6, AdamW updates to bf16 weights round to zero, so the model would train for hours and not move. Weights stay fp32 and compute runs under bf16 autocast; the config now refuses bf16 weights.",
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
      "Most of the work was the box, not the model. RunPod hosts with drivers older than CUDA 13 can't run the locked torch build, so setup now fails fast. Python block-buffers stdout when piped into `tee`, so the log looked frozen for minutes. And the pod's own ID lives in a file non-interactive shells skip, so the self-stop at the end ran with an empty ID until that was fixed.",
    ],
    figures: [{ value: "+4.8 pts", label: "one seed, 500-problem slice, step 0 to 300" }],
    caveat:
      "One seed on a 500-problem slice has a standard error of about 2.2 points. That is a sign of life, not a result. Turning it into one was M3's job.",
    links: [PR(1), { label: "The run", href: "/run?id=gsm8k-grpo-20261124-122830-3d7458" }],
  },
  {
    id: "three-seeds",
    date: "2026-11-24",
    milestone: "M3 · Evaluation",
    title: "+6.0 points, and the gain shrinks as you sample more",
    body: [
      "Three seeds, each scored on all 1,319 GSM8K test problems at its final checkpoint, against the base model on the same problems. Greedy accuracy went from 47.6% to 53.6%. The base and trained intervals don't overlap, and each seed beats base on its own (paired McNemar p ≤ 1.1e-05).",
      "pass@k tells the more interesting story. The gain is largest at pass@1 (+15.3 points) and falls to +6.3 at pass@16. Most of what GRPO does here is make the model reliably produce answers it could already sample, rather than teach it problems it could never solve.",
    ],
    figures: [
      { value: "47.6% → 53.6%", label: "greedy, full test set, mean of 3 seeds" },
      { value: "+6.0", label: "points, 95% CI [+4.7, +7.3] across seeds" },
      { value: "+15.3 → +6.3", label: "gain at pass@1 vs pass@16" },
    ],
    caveat:
      "The in-training eval curve is on test problems, so picking the best-looking step would inflate the result. The headline scores the final checkpoint, always.",
    links: [PR(2), { label: "Compare", href: "/compare" }],
  },
  {
    id: "dashboard",
    date: "2026-11-25",
    milestone: "M4 · Dashboard",
    title: "The run store is the only contract, so the dashboard needs no GPU",
    body: [
      "Trainer and dashboard never talk. The trainer writes SQLite metrics and Parquet rollouts; a read-only API serves them; `export-static` snapshots every response to JSON; the site is a static export that reads either source. That is what lets the dashboard be built and hosted while the GPU is off.",
      "Three views: Runs (final metrics and the config keys that differ), Run detail (curves, click through to the samples behind them) and the Rollout inspector (all eight samples of a prompt with their rewards and advantages, math rendered).",
    ],
    caveat:
      "GSM8K writes prices as `$3`, so a single `$` can't mean math. The inspector renders `\\(…\\)`, `\\[…\\]` and `$$…$$` only.",
    links: [PR(4), PR(5)],
  },
  {
    id: "mvp",
    date: "2026-11-26",
    milestone: "MVP · Close-out",
    title: "The dashboard went live, and was one merge from being replaced by its own API",
    body: [
      "Closing the MVP meant a clean-clone reproduction (locked install, tests, the toy run and a GSM8K smoke run, each recording its commit), the result figure, and deploying the dashboard: a prebuilt upload to Vercel, because the snapshot comes from a local run store that isn't in git.",
      "Linking the Vercel project quietly connected it to the GitHub repo. Its git build auto-detected the FastAPI app and deployed it, with no data, as the preview for the close-out PR. Merging would have done the same to production. A root `vercel.json` now turns git deploys off.",
    ],
    caveat:
      "Checking that protected preview with `vercel curl` silently added a deployment-protection bypass secret to the project. It was revoked the same evening.",
    links: [PR(6), { label: "Live dashboard", href: "https://grpo-lab-seven.vercel.app" }],
  },
  {
    id: "token-traces",
    date: "2026-11-26",
    milestone: "Token traces",
    title: "Per-token traces without a single extra forward pass",
    body: [
      "On every step that logs rollouts, the trainer now also keeps per-token log-prob, entropy, KL and ratio for the first two groups, copied from tensors the update already computes. The inspector colours each completion by any of them on one scale per group, so the eight siblings compare token by token.",
      "The check that matters: on the toy task, traced entropy and KL averaged over every token equal the step's logged metrics, which rules out an off-by-one or a wrong truncation. On a real GSM8K run, joining the per-token strings reproduced every traced completion exactly.",
    ],
    figures: [
      { value: "0", label: "extra forward passes" },
      { value: "~7 MB", label: "snapshot JSON per 300-step run, projected from ~50 bytes a token" },
    ],
    caveat:
      "With one optimisation pass per batch the ratio is exactly 1, which is true of every run so far, so that view stays disabled until a run uses μ > 1. The deployed runs predate traces and have no heatmap.",
    links: [PR(7)],
  },
  {
    id: "compare-ux",
    date: "2026-11-26",
    milestone: "Dashboard · Compare and usability",
    title: "Seeds pooled, and each config named by what it changes",
    body: [
      "The Compare page groups runs by training config, pools their seeds into a mean line with a 95% t-band, and labels every group by its difference from the baseline, so an ablation reads as `algo.loss_aggregation=constant` rather than a run name.",
      "A usability pass followed. On a phone the rollout-step ticks were 12 px wide and 10 px apart, so they became a step picker. The step slider announced its index, “15”, rather than “Step 150 of 300”. A bad link showed a bare 404 with no way back.",
    ],
    caveat:
      "Logging settings are ignored when grouping. The new runs record `log.trace_groups`, which the M3 runs predate, and without that rule logging would have looked like an ablation.",
    links: [{ label: "Compare", href: "/compare" }],
  },
  {
    id: "m7",
    date: "2026-11-26",
    milestone: "M7 · RQ1 ablations",
    status: "in-progress",
    title: "Two published recipes against the baseline, two seeds each",
    body: [
      "Which GRPO design choices matter at this scale? Dr. GRPO drops the per-group std scaling and uses a constant loss normaliser. DAPO-lite uses a token-level mean, clip-higher at 0.28 and no KL. Each runs at seeds 0 and 1 on its own pod at the same time. M3's three seeds are the baseline, since that config is unchanged.",
      "Every run ends with the same full-test-set greedy eval as M3, so the comparison uses the same yardstick. Results land here when the pods finish.",
    ],
    figures: [
      { value: "4", label: "runs in parallel: 3 × A40, 1 × L40S" },
      { value: "$2.86/hr", label: "for all four, each stopping itself when done" },
    ],
    caveat:
      "Two seeds make for wide intervals: the t-multiplier is 12.7. A third seed of each is the cheapest way to tighten them. DAPO-lite also lacks dynamic sampling and the overlong penalty, which aren't implemented yet.",
    links: [{ label: "Compare", href: "/compare" }],
  },
];
