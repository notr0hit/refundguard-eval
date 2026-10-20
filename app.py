"""Streamlit entrypoint for RefundGuard-Eval."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ui.components import render_header
from ui.pages import (
    render_failure_explorer_tab,
    render_rlhf_tab,
    render_sandbox_tab,
    render_scorecard_tab,
)
from ui.styles import CUSTOM_CSS

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = ROOT_DIR / "results" / "raw_runs.json"
PREFS_PATH = ROOT_DIR / "annotation" / "rlhf_preferences.jsonl"

load_dotenv(ROOT_DIR / ".env")
for _key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
    if not os.getenv(_key):
        try:
            _val = str(st.secrets.get(_key, "")).strip()
            if _val:
                os.environ[_key] = _val
        except Exception:
            pass

st.set_page_config(
    page_title="RefundGuard-Eval | Policy & Hallucination Auditor",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_header(
    "RefundGuard-Eval",
    "Policy compliance benchmark, contradiction auditor, and blinded RLHF preference explorer",
)

runs_df = (
    pd.DataFrame(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
    if RESULTS_PATH.exists()
    else pd.DataFrame()
)

tab_score, tab_explore, tab_rlhf, tab_sandbox = st.tabs(
    ["Scorecard", "Failure Audit", "RLHF Preferences", "Policy Sandbox"]
)

with tab_score:
    render_scorecard_tab(runs_df)

with tab_explore:
    render_failure_explorer_tab(runs_df)

with tab_rlhf:
    render_rlhf_tab(PREFS_PATH, runs_df)

with tab_sandbox:
    render_sandbox_tab()
