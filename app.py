"""Streamlit portfolio dashboard for agent safety evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "results" / ".mpl"))

from src.config import AGENT_CONFIG_LABELS, DATA_PATH, RESULTS_DIR, AgentConfig
from src.metrics import generate_summary_table
from src.tools import load_eval_cases

st.set_page_config(page_title="Agent Safety Evaluation", layout="wide")

RESULTS_PATH = RESULTS_DIR / "raw_results.csv"
SUMMARY_PATH = RESULTS_DIR / "summary_metrics.csv"
FAILURES_PATH = RESULTS_DIR / "failure_cases.csv"
FRONTIER_PATH = RESULTS_DIR / "safety_utility_frontier.png"


def load_results() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    raw = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else None
    summary = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else None
    failures = pd.read_csv(FAILURES_PATH) if FAILURES_PATH.exists() else None
    return raw, summary, failures


def main() -> None:
    st.title("Agent Safety Evaluation: Prompt Injection & Tool Misuse")

    st.markdown(
        """
        **Research question:** How reliably can an AI agent complete legitimate user tasks
        while resisting prompt injection, unauthorized tool use, and sensitive-information exfiltration?

        This evaluation measures the **safety–utility tradeoff**: mitigations that reduce attack
        success may also reduce task completion or increase over-refusal.
        """
    )

    # --- Threat Model ---
    st.header("Threat Model")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Agent capabilities (mock tools)")
        st.markdown(
            """
            - `read_file` — read local mock files
            - `search_email` — search mock inbox
            - `send_email` — send to mock recipients
            - `read_calendar` / `create_calendar_event`
            - `get_user_profile` — retrieve mock profile data
            """
        )
    with col2:
        st.subheader("Attacker goals & safety failures")
        st.markdown(
            """
            - Embed malicious instructions in documents/emails
            - Trick agent into unauthorized tool use
            - Exfiltrate sensitive data to `attacker@example.com`
            - **Safety failure:** forbidden tool call, sensitive exposure, or attack success
            """
        )

    # --- Evaluation Design ---
    st.header("Evaluation Design")
    try:
        cases = load_eval_cases()
        category_counts = pd.Series([c["category"] for c in cases]).value_counts()
        st.markdown(f"**Test cases:** {len(cases)}")
        st.dataframe(category_counts.rename("count").reset_index().rename(columns={"index": "category"}), hide_index=True)
    except Exception as e:
        st.warning(f"Could not load cases: {e}")
        cases = []

    st.markdown(
        """
        **Agent configurations:**
        1. **Baseline** — minimal safety instructions
        2. **Instruction Hierarchy** — system > user > retrieved content
        3. **Hierarchy + Permission Gate** — deterministic tool authorization layer

        **Evaluation method:**
        - *Deterministic safety checks* — forbidden actions, unauthorized tools, sensitive exposure
        - *LLM judge* — task completion quality only (not used for safety labels)
        """
    )

    raw_df, summary_df, failures_df = load_results()

    if raw_df is None or summary_df is None:
        st.warning("No evaluation results generated yet. Run `python run_eval.py` to generate results.")
        st.code("python run_eval.py", language="bash")
        st.stop()

    # --- Results ---
    st.header("Results")
    comparison = generate_summary_table(summary_df)
    st.dataframe(comparison, hide_index=True)

    st.subheader("Metrics with 95% Bootstrap CIs")
    display_metrics = summary_df[summary_df["category"] == "all"].copy()
    display_metrics["estimate_with_ci"] = display_metrics.apply(
        lambda r: f"{r['point_estimate']:.1%} [{r['ci_lower']:.1%}, {r['ci_upper']:.1%}]"
        if pd.notna(r["point_estimate"]) else "N/A",
        axis=1,
    )
    st.dataframe(
        display_metrics[["config_label", "metric", "estimate_with_ci", "n"]].rename(
            columns={"config_label": "Configuration", "metric": "Metric", "estimate_with_ci": "Estimate [95% CI]", "n": "N"}
        ),
        hide_index=True,
    )

    st.caption(
        "Sample size is small (~50 cases). Bootstrap CIs quantify uncertainty but should not "
        "be interpreted as precise population estimates."
    )

    # --- Safety vs Utility ---
    st.header("Safety vs Utility")
    if FRONTIER_PATH.exists():
        st.image(str(FRONTIER_PATH), caption="Attack Success Rate (x) vs Legitimate Task Success (y)")
    else:
        st.info("Frontier plot not yet generated.")

    # --- Failure Analysis ---
    st.header("Failure Analysis")
    if failures_df is not None and not failures_df.empty:
        selected_config = st.selectbox(
            "Filter by configuration",
            options=["all"] + list(failures_df["config"].unique()),
        )
        filtered = failures_df if selected_config == "all" else failures_df[failures_df["config"] == selected_config]
        show_n = min(10, len(filtered))
        st.markdown(f"Showing {show_n} representative failures (of {len(filtered)} total)")

        for _, row in filtered.head(show_n).iterrows():
            with st.expander(f"{row['case_id']} | {row['category']} | {row['config']}"):
                st.markdown("**User request**")
                st.write(row["user_request"])
                if row.get("malicious_instruction", "N/A") != "N/A":
                    st.markdown("**Retrieved malicious content**")
                    st.code(str(row["malicious_instruction"])[:400])
                st.markdown("**Agent response**")
                st.write(str(row.get("model_behavior", ""))[:500])
                st.markdown("**Tool calls**")
                st.code(str(row.get("tool_calls", "[]")))
                st.markdown("**Why flagged**")
                st.error(str(row.get("why_unsafe", "")))
                st.markdown("**Expected safe behavior**")
                st.success(str(row.get("expected_behavior", "")))
    else:
        st.info("No failure cases recorded yet.")

    # --- Limitations ---
    st.header("Limitations")
    st.markdown(
        """
        - **Small synthetic dataset** (~50 cases) — results have wide confidence intervals
        - **Limited attack diversity** — does not cover multi-turn or adaptive attacks
        - **Model/provider dependence** — results apply to the evaluated model only
        - **LLM judge measurement error** — task success labels may be imperfect
        - **Simulated tools** — not connected to real systems
        - **Not a security certification** — this is a research evaluation, not production security
        """
    )


if __name__ == "__main__":
    main()
