"""README result figure: base vs GRPO on the full GSM8K test set, from `evaluate compare --json`.

    uv run evaluate compare evals/base evals/seed0 evals/seed1 evals/seed2 \
        --json report/gsm8k-3seed.json
    uv run --group report python scripts/report_figures.py

The comparison JSON is committed (evals/ is gitignored), so the figure can be rebuilt from a
clean checkout without the eval outputs or a GPU.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = "#eb6834"  # categorical slot 2 (orange): the reference model
GRPO = "#2a78d6"  # categorical slot 1 (blue): the trained policy
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def pct(ci: dict) -> tuple[float, float, float]:
    return 100 * ci["mean"], 100 * ci["low"], 100 * ci["high"]


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def greedy_panel(ax: plt.Axes, d: dict) -> None:
    rows = [("Base", pct(d["base"]["greedy_correct"]), BASE)]
    for i, run in enumerate(d["runs"]):
        rows.append((f"GRPO seed {i}", pct(run["greedy_correct"]), GRPO))
    rows.append(("GRPO, 3 seeds", pct(d["across_seeds"]["greedy_correct"]["trained"]), GRPO))
    rows.reverse()  # top-to-bottom reading order

    for y, (label, (m, lo, hi), color) in enumerate(rows):
        bold = label.endswith("seeds") or label == "Base"
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(m, y, "o", color=color, markersize=8 if bold else 6, markeredgecolor=SURFACE)
        ax.text(hi + 0.6, y, f"{m:.1f}", va="center", fontsize=9, color=INK)

    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    for tick, (label, _, _) in zip(ax.get_yticklabels(), rows, strict=True):
        if label.endswith("seeds") or label == "Base":
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Greedy accuracy, % (95% CI)", color=INK_2, fontsize=9)
    diff = pct(d["across_seeds"]["greedy_correct"]["diff"])
    ax.set_title(
        f"Greedy: +{diff[0]:.1f} pts [{diff[1]:+.1f}, {diff[2]:+.1f}]",
        loc="left",
        fontsize=11,
        color=INK,
    )


def pass_at_k_panel(ax: plt.Axes, d: dict) -> None:
    ks = sorted(int(k.split("@")[1]) for k in d["base"] if k.startswith("pass@"))
    series = [
        ("Base", [pct(d["base"][f"pass@{k}"]) for k in ks], BASE),
        ("GRPO, 3 seeds", [pct(d["across_seeds"][f"pass@{k}"]["trained"]) for k in ks], GRPO),
    ]
    for label, vals, color in series:
        m, lo, hi = zip(*vals, strict=True)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(ks, m, color=color, linewidth=2, marker="o", markersize=6)
        ax.text(ks[-1] * 1.12, m[-1], label, va="center", fontsize=9, color=INK)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(ks[0] * 0.85, ks[-1] * 2.6)
    ax.set_xlabel("k (samples at T = 1)", color=INK_2, fontsize=9)
    ax.set_ylabel("pass@k, %", color=INK_2, fontsize=9)
    ax.set_title("pass@k: the gain shrinks as k grows", loc="left", fontsize=11, color=INK)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--compare", type=Path, default=Path("report/gsm8k-3seed.json"))
    p.add_argument("--out", type=Path, default=Path("report/gsm8k-3seed.png"))
    args = p.parse_args()

    d = json.loads(args.compare.read_text())
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]}, facecolor=SURFACE
    )
    for ax in (a, b):
        style(ax)
    greedy_panel(a, d)
    pass_at_k_panel(b, d)
    fig.suptitle(
        f"Qwen2.5-0.5B-Instruct, GRPO vs base: all {d['num_problems']} GSM8K test problems",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
