#!/usr/bin/env python3
"""Run agent safety evaluation across cases and configurations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import AgentRunner
from src.config import (
    EVALUATOR_VERSION,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RESULTS_DIR,
    AgentConfig,
)
from src.evaluator import evaluate_run
from src.metrics import process_results
from src.tools import load_eval_cases


def run_evaluation(
    limit: int | None = None,
    config_filter: AgentConfig | None = None,
    use_llm_judge: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run evaluation and return results DataFrame."""
    cases = load_eval_cases()
    if limit:
        cases = cases[:limit]

    configs = [config_filter] if config_filter else AgentConfig.all_configs()
    rows: list[dict] = []

    for config in configs:
        runner = AgentRunner(config=config)
        for case in cases:
            print(f"Running {case['case_id']} | config={config.value}")

            latency_ms = 0.0
            if dry_run:
                from src.tools import MockToolEnvironment

                trace = MockToolEnvironment(case.get("environment", {})).trace
                final_response = "[dry-run: no agent execution]"
            else:
                result = runner.run(case)
                trace = result.tool_trace
                final_response = result.final_response
                latency_ms = result.latency_ms

            eval_result = evaluate_run(
                case, final_response, trace, use_llm_judge=use_llm_judge and not dry_run
            )

            rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluator_version": EVALUATOR_VERSION,
                "model": OPENAI_MODEL if OPENAI_API_KEY else "mock",
                "config": config.value,
                "case_id": case["case_id"],
                "category": case["category"],
                "is_attack": case.get("is_attack", False),
                "is_benign_control": case.get("is_benign_control", False),
                "user_request": case["user_request"],
                "final_response": final_response,
                "tools_requested": json.dumps([r.tool_name for r in trace.requested]),
                "tools_executed": json.dumps([
                    {"tool": r.tool_name, "args": r.arguments} for r in trace.executed
                ]),
                "tools_blocked": json.dumps([
                    {"tool": r.tool_name, "args": r.arguments, "reason": r.block_reason}
                    for r in trace.blocked
                ]),
                "refused": eval_result.refused,
                "forbidden_action_occurred": eval_result.safety.forbidden_action_occurred,
                "attack_success": eval_result.safety.attack_success,
                "unauthorized_tool_call": eval_result.safety.unauthorized_tool_call,
                "sensitive_exposed": eval_result.safety.sensitive_exposed,
                "safety_violations": "; ".join(eval_result.safety.safety_violations),
                "task_completed": eval_result.quality.task_completed,
                "task_completed_source": eval_result.quality.source,
                "judge_rationale": eval_result.quality.judge_rationale,
                "latency_ms": latency_ms,
                "sensitive_markers_present": bool(case.get("sensitive_markers")),
            })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "raw_results.csv", index=False)
    print(f"\nWrote {len(df)} results to {RESULTS_DIR / 'raw_results.csv'}")

    _, metrics_df = process_results()
    if not metrics_df.empty:
        print(f"Wrote summary metrics to {RESULTS_DIR / 'summary_metrics.csv'}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent safety evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument(
        "--config",
        type=str,
        choices=[c.value for c in AgentConfig],
        default=None,
        help="Run single configuration",
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--dry-run", action="store_true", help="Skip agent execution")
    args = parser.parse_args()

    config_filter = AgentConfig(args.config) if args.config else None

    if not OPENAI_API_KEY and not args.dry_run:
        print("Warning: OPENAI_API_KEY not set. Agent will return mock responses.")

    run_evaluation(
        limit=args.limit,
        config_filter=config_filter,
        use_llm_judge=not args.no_judge,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
