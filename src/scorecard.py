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


ATTACK_LABELS = {
    "boundary_escalation": "Boundary Escalation",
    "fake_policy": "Fake Policy",
    "prompt_injection": "Prompt Injection",
    "sycophancy_pressure": "Sycophancy Pressure",
    "valid_exception": "Valid Exception",
}


def render_chart(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Render a clean, publication-grade 2-panel scorecard chart to results/scorecard.png."""
    import numpy as np

    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="#FFFFFF", gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel 1: Horizontal bar chart (Decision Acc, Clause Hit, Contradiction) per Model
    models = summary["Model"].tolist()[::-1]
    y = np.arange(len(models))
    h = 0.25

    acc_vals = summary["Decision Acc (%)"].tolist()[::-1]
    clause_vals = summary["Clause Hit (%)"].tolist()[::-1]
    contra_vals = summary["Contradiction (%)"].tolist()[::-1]

    b1 = axes[0].barh(y + h, acc_vals, h, label="Decision Acc (%)", color="#0F172A", zorder=3)
    b2 = axes[0].barh(y, clause_vals, h, label="Clause Hit (%)", color="#2563EB", zorder=3)
    b3 = axes[0].barh(y - h, contra_vals, h, label="Contradiction (%)", color="#DC2626", zorder=3)

    for bar in list(b1) + list(b2) + list(b3):
        w = bar.get_width()
        axes[0].text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=8, color="#334155")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(models, fontsize=9, color="#0F172A")
    axes[0].set_xlim(0, 118)
    axes[0].set_xlabel("Percentage (%)", fontsize=9, color="#475569")
    axes[0].set_title("Model Compliance & Contradiction", fontsize=11, fontweight="bold", color="#0F172A", pad=34)
    axes[0].grid(axis="x", linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.09), ncol=3, frameon=False, fontsize=8)

    # Panel 2: Annotated Heatmap of Accuracy by Attack Category x Model
    pivot = (
        df.pivot_table(index="attack_type", columns="model", values="decision_match", aggfunc="mean")
        * 100
    )
    row_labels = [ATTACK_LABELS.get(k, k) for k in pivot.index]
    col_labels = [c.split("/")[-1] for c in pivot.columns]
    matrix = pivot.to_numpy(dtype=float)

    cmap = plt.cm.Blues
    masked = np.ma.masked_invalid(matrix)
    cmap.set_bad(color="#F1F5F9")
    axes[1].imshow(masked, cmap=cmap, vmin=40, vmax=100, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                txt, col = "—", "#94A3B8"
            else:
                txt = f"{val:.0f}%"
                col = "#FFFFFF" if val >= 80 else "#0F172A"
            axes[1].text(j, i, txt, ha="center", va="center", fontsize=9.5, fontweight="bold", color=col)

    axes[1].set_xticks(np.arange(len(col_labels)))
    axes[1].set_xticklabels(col_labels, fontsize=9, color="#0F172A", rotation=12, ha="right")
    axes[1].set_yticks(np.arange(len(row_labels)))
    axes[1].set_yticklabels(row_labels, fontsize=9, color="#0F172A")
    axes[1].set_title("Decision Accuracy by Attack Category", fontsize=11, fontweight="bold", color="#0F172A", pad=14)
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#E2E8F0")

    plt.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=200, facecolor="#FFFFFF", bbox_inches="tight")
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

