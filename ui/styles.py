"""Restrained, production-style CSS for the RefundGuard-Eval dashboard."""

CUSTOM_CSS = """
<style>
    /* Tighten default Streamlit padding for a denser, tool-like layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Clean header bar */
    .rg-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .rg-title {
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin: 0;
    }
    .rg-subtitle {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 0.25rem;
    }

    /* Metric cards */
    .rg-kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
    }
    .rg-kpi-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .rg-kpi-value {
        font-size: 1.45rem;
        font-weight: 600;
        color: #0F172A;
        font-variant-numeric: tabular-nums;
    }
    .rg-kpi-note {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Subtle status badges */
    .rg-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 4px;
        letter-spacing: 0.03em;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .rg-badge-pass {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .rg-badge-fail {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .rg-badge-warn {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .rg-badge-neutral {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
    }

    /* Case inspection box */
    .rg-response-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #0F172A;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #1E293B;
        margin: 0.5rem 0;
    }

    /* Hide default Streamlit branding clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

