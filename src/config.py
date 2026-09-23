"""Central configuration for the agent safety evaluation framework."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "eval_cases.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42
EVALUATOR_VERSION = "0.1.0"
MAX_TOOL_ROUNDS = 5
BOOTSTRAP_SAMPLES = 1000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", OPENAI_MODEL)

AUTHORIZED_EMAIL_RECIPIENTS = {"user@example.com", "team@example.com", "manager@example.com"}


class AgentConfig(str, Enum):
    BASELINE = "baseline"
    INSTRUCTION_HIERARCHY = "instruction_hierarchy"
    HIERARCHY_PERMISSION_GATE = "hierarchy_permission_gate"

    @classmethod
    def all_configs(cls) -> list[AgentConfig]:
        return list(cls)


AGENT_CONFIG_LABELS = {
    AgentConfig.BASELINE: "Baseline",
    AgentConfig.INSTRUCTION_HIERARCHY: "Instruction Hierarchy",
    AgentConfig.HIERARCHY_PERMISSION_GATE: "Hierarchy + Permission Gate",
}
