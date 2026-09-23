"""Metrics computation, bootstrap CIs, and failure analysis export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AGENT_CONFIG_LABELS, RANDOM_SEED, RESULTS_DIR, AgentConfig
from src.evaluator import extract_malicious_instruction


BOOTSTRAP_SAMPLES = 1000


def bootstrap_ci(values: pd.Series, n_samples: int = BOOTSTRAP_SAMPLES, seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """Compute point estimate and 95% bootstrap CI for a binary metric."""
    arr = values.dropna().astype(float).values
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(arr.mean())
    if n == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_samples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return point, ci_lower, ci_upper


def compute_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all metrics with bootstrap CIs per config and category."""
    rows: list[dict[str, Any]] = []

    metric_defs = [
        ("attack_success_rate", lambda df: df[df["is_attack"]]["attack_success"]),
        ("legitimate_task_success", lambda df: df[df["task_completed"].notna()]["task_completed"]),
        ("over_refusal_rate", lambda df: df[df["is_benign_control"]]["refused"]),
        ("unauthorized_tool_call_rate", lambda df: df["unauthorized_tool_call"]),
        ("sensitive_exposure_rate", lambda df: df[(df["is_attack"]) & (df["sensitive_markers_present"])]["sensitive_exposed"]),
    ]

    for config in raw_df["config"].unique():
        config_df = raw_df[raw_df["config"] == config]

        for metric_name, selector in metric_defs:
            values = selector(config_df)
            point, ci_lower, ci_upper = bootstrap_ci(values)
            rows.append({
                "metric": metric_name,
                "config": config,
                "config_label": AGENT_CONFIG_LABELS.get(AgentConfig(config), config),
                "category": "all",
                "point_estimate": point,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(values.dropna()),
            })

        for category in config_df["category"].unique():
            cat_df = config_df[config_df["category"] == category]
            if cat_df["is_attack"].any():
                values = cat_df[cat_df["is_attack"]]["attack_success"]
                point, ci_lower, ci_upper = bootstrap_ci(values)
                rows.append({
                    "metric": "attack_success_rate",
                    "config": config,
                    "config_label": AGENT_CONFIG_LABELS.get(AgentConfig(config), config),
                    "category": category,
                    "point_estimate": point,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n": len(values.dropna()),
                })

    return pd.DataFrame(rows)


def _load_cases_by_id() -> dict[str, dict]:
    from src.tools import load_eval_cases
    return {c["case_id"]: c for c in load_eval_cases()}


def export_failure_cases(raw_df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Export representative failure cases for qualitative analysis."""
    cases_by_id = _load_cases_by_id()
    failures = raw_df[
        (raw_df["attack_success"] == True)  # noqa: E712
        | ((raw_df["is_benign_control"] == True) & (raw_df["refused"] == True))  # noqa: E712
        | ((raw_df["is_attack"] == False) & (raw_df["task_completed"] == False))  # noqa: E712
    ].copy()

    rows = []
    for _, row in failures.iterrows():
        case = cases_by_id.get(row["case_id"], {})
        why_unsafe = row.get("safety_violations", "") or (
            "Over-refusal on benign request" if row.get("refused") and row.get("is_benign_control") else
            "Task not completed" if row.get("task_completed") == False else "Unknown"  # noqa: E712
        )
        rows.append({
            "case_id": row["case_id"],
            "category": row["category"],
            "config": row["config"],
            "user_request": row["user_request"],
            "malicious_instruction": extract_malicious_instruction(case),
            "model_behavior": str(row.get("final_response", ""))[:500],
            "tool_calls": row.get("tools_executed", ""),
            "why_unsafe": why_unsafe,
            "expected_behavior": case.get("expected_behavior", ""),
        })

    failure_df = pd.DataFrame(rows)
    out = output_path or RESULTS_DIR / "failure_cases.csv"
    if not failure_df.empty:
        failure_df.to_csv(out, index=False)
    else:
        pd.DataFrame(columns=[
            "case_id", "category", "config", "user_request", "malicious_instruction",
            "model_behavior", "tool_calls", "why_unsafe", "expected_behavior",
        ]).to_csv(out, index=False)
    return failure_df


def generate_frontier_plot(raw_df: pd.DataFrame, output_path: Path | None = None) -> Path | None:
    """Generate safety-utility tradeoff visualization."""
    out = output_path or RESULTS_DIR / "safety_utility_frontier.png"

    summary_rows = []
    for config in raw_df["config"].unique():
        cdf = raw_df[raw_df["config"] == config]
        asr = cdf[cdf["is_attack"]]["attack_success"].mean() if cdf["is_attack"].any() else float("nan")
        task_success = cdf[cdf["task_completed"].notna()]["task_completed"].mean()
        label = AGENT_CONFIG_LABELS.get(AgentConfig(config), config)
        summary_rows.append({"config": config, "label": label, "asr": asr, "task_success": task_success})

    if not summary_rows or all(np.isnan(r["asr"]) for r in summary_rows):
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in summary_rows:
        if not np.isnan(row["asr"]) and not np.isnan(row["task_success"]):
            ax.scatter(row["asr"], row["task_success"], s=120, zorder=3)
            ax.annotate(row["label"], (row["asr"], row["task_success"]),
                        textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Attack Success Rate (lower = safer)")
    ax.set_ylabel("Legitimate Task Success Rate (higher = better)")
    ax.set_title("Safety vs Utility Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot metrics into comparison table for dashboard."""
    pivot_rows = []
    for config in metrics_df[metrics_df["category"] == "all"]["config"].unique():
        cdf = metrics_df[(metrics_df["config"] == config) & (metrics_df["category"] == "all")]
        row = {"Configuration": AGENT_CONFIG_LABELS.get(AgentConfig(config), config)}
        for _, m in cdf.iterrows():
            col_name = m["metric"].replace("_", " ").title()
            row[col_name] = f"{m['point_estimate']:.2%}" if not np.isnan(m["point_estimate"]) else "N/A"
            if not np.isnan(m["ci_lower"]):
                row[f"{col_name} CI"] = f"[{m['ci_lower']:.2%}, {m['ci_upper']:.2%}]"
        pivot_rows.append(row)
    return pd.DataFrame(pivot_rows)


def process_results(raw_results_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process raw results into summary metrics and failure cases."""
    path = raw_results_path or RESULTS_DIR / "raw_results.csv"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.read_csv(path)
    # Flag cases with sensitive markers for exposure rate denominator
    raw_df["sensitive_markers_present"] = raw_df.get("sensitive_markers_present", True)

    metrics_df = compute_metrics(raw_df)
    metrics_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    export_failure_cases(raw_df)
    generate_frontier_plot(raw_df)
    return raw_df, metrics_df
