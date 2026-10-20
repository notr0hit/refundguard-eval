"""Reusable UI components for RefundGuard-Eval."""

import html
import streamlit as st


def render_header(title: str, subtitle: str) -> None:
    """Render a clean, restrained page header."""
    st.markdown(
        f"""
        <div class="rg-header">
            <div class="rg-title">{html.escape(title)}</div>
            <div class="rg-subtitle">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, note: str = "") -> None:
    """Render a compact KPI metric card with tabular numerals."""
    note_html = f'<div class="rg-kpi-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="rg-kpi-card">
            <div class="rg-kpi-label">{html.escape(label)}</div>
            <div class="rg-kpi-value">{html.escape(value)}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(text: str, variant: str = "neutral") -> str:
    """Return HTML for a subtle monospace status badge."""
    allowed = {"pass", "fail", "warn", "neutral"}
    cls = variant if variant in allowed else "neutral"
    return f'<span class="rg-badge rg-badge-{cls}">{html.escape(text)}</span>'


def render_response_box(text: str) -> None:
    """Render model customer-facing output in a clean bordered block."""
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(f'<div class="rg-response-box">{safe}</div>', unsafe_allow_html=True)

