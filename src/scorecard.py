"""Generate benchmark summary metrics, Markdown table, and results/scorecard.png."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT_DIR / "results" / "raw_runs.json"
PREFS_PATH = ROOT_DIR / "annotation" / "rlhf_preferences.jsonl"
CHART_PATH = ROOT_DIR / "results" / "scorecard.png"


def load_rlhf_stats() -> dict[str, dict[str, float]]:
    """Aggregate human RLHF scores and win rates per model if preferences exist."""
    if not PREFS_PATH.exists():
        return {}
    scores: dict[str, list[int]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)
    appearances: dict[str, int] = defaultdict(int)

    with PREFS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ma, mb = row["model_a"], row["model_b"]
            scores[ma].append(row["score_a"])
            scores[mb].append(row["score_b"])
            appearances[ma] += 1
            appearances[mb] += 1
            if row["winner_model"] == ma:
                wins[ma] += 1
            elif row["winner_model"] == mb:
                wins[mb] += 1

    stats = {}
    for model, s_list in scores.items():
        stats[model] = {
            "avg_rlhf": round(sum(s_list) / len(s_list), 2),
            "win_rate": round(100 * wins[model] / appearances[model], 1),
        }
    return stats


def build_summary(df: pd.DataFrame, rlhf: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Compute per-model evaluation metrics."""
    rows = []
    for model, grp in df.groupby("model"):
        en_grp = grp[grp["language"] == "en"]
        hi_grp = grp[grp["language"] == "hinglish"]
        model_rlhf = rlhf.get(model, {})
        rows.append({
            "Model": model,
            "Cases": len(grp),
            "Decision Acc (%)": round(100 * grp["decision_match"].mean(), 1),
            "Clause Hit (%)": round(100 * grp["clause_hit"].mean(), 1),
            "Contradiction (%)": round(100 * grp["contradiction"].mean(), 1),
            "EN Acc (%)": round(100 * en_grp["decision_match"].mean(), 1) if len(en_grp) else "-",
            "Hinglish Acc (%)": round(100 * hi_grp["decision_match"].mean(), 1) if len(hi_grp) else "-",
            "RLHF Score (1-5)": model_rlhf.get("avg_rlhf", "-"),
            "RLHF Win (%)": model_rlhf.get("win_rate", "-"),
        })
    return pd.DataFrame(rows).sort_values("Decision Acc (%)", ascending=False)


def render_chart(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Render a clean, editorial 2-panel scorecard chart to results/scorecard.png."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor="#FFFFFF")

    # Panel 1: Decision Accuracy vs Contradiction Rate by Model
    models = summary["Model"].tolist()
    x = list(range(len(models)))
    width = 0.34
    acc_vals = summary["Decision Acc (%)"].tolist()
    contra_vals = summary["Contradiction (%)"].tolist()

    bars1 = axes[0].bar([i - width / 2 for i in x], acc_vals, width, label="Decision Accuracy (%)", color="#0F172A", zorder=3)
    bars2 = axes[0].bar([i + width / 2 for i in x], contra_vals, width, label="Contradiction Rate (%)", color="#DC2626", zorder=3)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 2.2, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.5, color="#334155")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9, color="#1E293B")
    axes[0].set_ylim(0, 118)
    axes[0].set_ylabel("Percentage (%)", fontsize=9.5, color="#475569")
    axes[0].set_title("Policy Accuracy vs. Contradiction Rate", fontsize=11, fontweight="bold", color="#0F172A", pad=26)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.11), ncol=2, frameon=False, fontsize=8.5)

    # Panel 2: Accuracy by Attack Type across Models
    pivot = (
        df.pivot_table(index="attack_type", columns="model", values="decision_match", aggfunc="mean")
        * 100
    )
    palette = ["#0F172A", "#2563EB", "#64748B", "#0EA5E9"]
    pivot.plot(
        kind="bar",
        ax=axes[1],
        color=palette[: len(pivot.columns)],
        width=0.68,
        zorder=3,
    )
    axes[1].set_ylim(0, 118)
    axes[1].set_ylabel("Decision Accuracy (%)", fontsize=9.5, color="#475569")
    axes[1].set_xlabel("")
    axes[1].set_title("Accuracy by Attack Category", fontsize=11, fontweight="bold", color="#0F172A", pad=26)
    axes[1].tick_params(axis="x", rotation=0, labelsize=9, colors="#1E293B")
    axes[1].grid(axis="y", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, 1.11), ncol=len(pivot.columns), frameon=False, fontsize=8)

    fig.suptitle("RefundGuard-Eval Benchmark Scorecard", fontsize=13.5, fontweight="bold", color="#0F172A", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=200, facecolor="#FFFFFF")
    plt.close(fig)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing {RESULTS_PATH}. Run `python src/run_eval.py` first.")

    df = pd.DataFrame(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
    rlhf = load_rlhf_stats()
    summary = build_summary(df, rlhf)
    render_chart(df, summary)

    print("\n### Benchmark Results (Paste into README.md)\n")
    print(summary.to_markdown(index=False))
    print(f"\nSaved scorecard chart to {CHART_PATH}")


if __name__ == "__main__":
    main()

