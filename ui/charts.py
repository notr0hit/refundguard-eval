"""Clean, interactive Altair visualizations for RefundGuard-Eval."""

import altair as alt
import pandas as pd

PALETTE = ["#0F172A", "#2563EB", "#64748B", "#0EA5E9"]


def _base_config(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="#E2E8F0",
            gridDash=[3, 3],
            domainColor="#CBD5E1",
            labelColor="#334155",
            titleColor="#475569",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            orient="top",
            title=None,
            labelColor="#334155",
            labelFontSize=11,
        )
    )


def build_language_chart(df: pd.DataFrame) -> alt.Chart:
    """Compare English vs. Hinglish decision accuracy across models."""
    lang_df = (
        df.groupby(["model", "language"], as_index=False)["decision_match"]
        .mean()
        .assign(
            accuracy=lambda d: (d["decision_match"] * 100).round(1),
            language_label=lambda d: d["language"].map({"en": "English", "hinglish": "Hinglish"}).fillna(d["language"]),
            short_model=lambda d: d["model"].str.split("/").str[-1],
        )
    )
    chart = (
        alt.Chart(lang_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("short_model:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("language_label:N"),
            y=alt.Y("accuracy:Q", title="Decision Accuracy (%)", scale=alt.Scale(domain=[0, 105])),
            color=alt.Color(
                "language_label:N",
                scale=alt.Scale(domain=["English", "Hinglish"], range=["#0F172A", "#2563EB"]),
            ),
            tooltip=["model:N", "language_label:N", alt.Tooltip("accuracy:Q", title="Accuracy (%)")],
        )
        .properties(height=260, title="English vs. Hinglish Accuracy")
    )
    return _base_config(chart)


def build_domain_chart(df: pd.DataFrame) -> alt.Chart:
    """Compare policy compliance across airline, ecommerce, and food domains."""
    dom_df = (
        df.groupby(["model", "domain"], as_index=False)["decision_match"]
        .mean()
        .assign(
            accuracy=lambda d: (d["decision_match"] * 100).round(1),
            domain_label=lambda d: d["domain"].str.title(),
            short_model=lambda d: d["model"].str.split("/").str[-1],
        )
    )
    chart = (
        alt.Chart(dom_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("domain_label:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("short_model:N"),
            y=alt.Y("accuracy:Q", title="Decision Accuracy (%)", scale=alt.Scale(domain=[0, 105])),
            color=alt.Color("short_model:N", scale=alt.Scale(range=PALETTE)),
            tooltip=["model:N", "domain_label:N", alt.Tooltip("accuracy:Q", title="Accuracy (%)")],
        )
        .properties(height=260, title="Accuracy by Policy Domain")
    )
    return _base_config(chart)


def build_confusion_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Render Expected vs. Predicted decision confusion matrix across all runs."""
    order = ["APPROVE", "DENY", "PARTIAL", "ESCALATE"]
    cm_df = (
        df[df["model_decision"].isin(order)]
        .groupby(["expected_decision", "model_decision"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    base = alt.Chart(cm_df).encode(
        x=alt.X("model_decision:N", title="Predicted Decision", sort=order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("expected_decision:N", title="Expected Decision", sort=order),
    )
    rect = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["expected_decision:N", "model_decision:N", "count:Q"],
    )
    text = base.mark_text(fontWeight="bold", fontSize=12).encode(
        text="count:Q",
        color=alt.condition(alt.datum.count > cm_df["count"].max() * 0.45, alt.value("#FFFFFF"), alt.value("#0F172A")),
    )
    return _base_config((rect + text).properties(height=260, title="Decision Confusion Matrix (All Models)"))


def build_latency_accuracy_chart(df: pd.DataFrame) -> alt.Chart:
    """Plot median latency vs. decision accuracy per model."""
    lat_df = (
        df.groupby("model", as_index=False)
        .agg(
            accuracy=("decision_match", lambda s: round(s.mean() * 100, 1)),
            median_ms=("latency_ms", lambda s: round(s.median(), 0)),
            cases=("case_id", "count"),
        )
        .assign(short_model=lambda d: d["model"].str.split("/").str[-1])
    )
    min_acc = max(0, float(lat_df["accuracy"].min()) - 8)
    points = (
        alt.Chart(lat_df)
        .mark_circle(size=160, opacity=0.9)
        .encode(
            x=alt.X("median_ms:Q", title="Median Latency (ms)"),
            y=alt.Y("accuracy:Q", title="Decision Accuracy (%)", scale=alt.Scale(domain=[min_acc, 102])),
            color=alt.Color("short_model:N", scale=alt.Scale(range=PALETTE), legend=None),
            tooltip=[
                "model:N",
                alt.Tooltip("accuracy:Q", title="Accuracy (%)"),
                alt.Tooltip("median_ms:Q", title="Median Latency (ms)"),
                "cases:Q",
            ],
        )
    )
    labels = (
        alt.Chart(lat_df)
        .mark_text(align="left", dx=10, dy=-4, fontSize=11, color="#0F172A")
        .encode(x="median_ms:Q", y="accuracy:Q", text="short_model:N")
    )
    return _base_config((points + labels).properties(height=260, title="Accuracy vs. Median Latency"))
