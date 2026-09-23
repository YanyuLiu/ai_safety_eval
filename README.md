# Agent Safety Evaluation: Prompt Injection & Tool Misuse

A lightweight evaluation framework measuring whether a tool-using AI agent can complete legitimate tasks while resisting prompt injection, unauthorized tool use, and sensitive-information exfiltration.

## Motivation

Tool-using agents can read files, search email, and send messages on a user's behalf. This creates a new attack surface: malicious instructions embedded in retrieved content can hijack agent behavior. Effective safety is not simply maximizing refusal—a useful agent must balance **safety**, **utility**, and **calibration**.

## Research Question

**How reliably can an AI agent complete legitimate user tasks while resisting prompt injection, unauthorized tool use, and sensitive-information exfiltration?**

## Threat Model

### Agent capabilities (mock, local only)
- `read_file`, `search_email`, `send_email`, `read_calendar`, `create_calendar_event`, `get_user_profile`

### Attacker capabilities
- Embed instructions in documents, emails, or tool outputs
- Request direct exfiltration via user messages
- Trick the agent into using unnecessary tools

### Safety failures
- Executing forbidden tool calls (e.g., `send_email` to `attacker@example.com`)
- Unauthorized tool use beyond task scope
- Exposing sensitive markers in responses or outbound emails

## Evaluation Dataset

**File:** `data/eval_cases.jsonl` (~50 cases)

| Category | Description |
|----------|-------------|
| `direct_prompt_injection` | User explicitly requests policy violations |
| `indirect_prompt_injection` | Malicious instructions embedded in retrieved content |
| `unauthorized_tool_use` | Task scoped to subset of tools; others forbidden |
| `sensitive_exfiltration` | Sensitive data + instruction to transmit externally |
| `benign_sensitive_looking` | Legitimate tasks with security-related language (over-refusal controls) |

Each case includes: `allowed_tools`, `forbidden_actions`, `sensitive_markers`, `expected_behavior`, and `task_success_criteria`.

## Experimental Design

Three agent configurations:

| Config | Description |
|--------|-------------|
| **A — Baseline** | Minimal system prompt |
| **B — Instruction Hierarchy** | System > user > retrieved content; untrusted data rules |
| **C — Hierarchy + Permission Gate** | Config B plus deterministic pre-execution tool authorization |

## Metrics

| Metric | Definition | Direction |
|--------|------------|-----------|
| Attack Success Rate (ASR) | Successful attacks / attack cases | Lower is better |
| Legitimate Task Success | Completed legitimate tasks / applicable cases | Higher is better |
| Over-Refusal Rate | Unnecessary refusals / benign controls | Lower is better |
| Unauthorized Tool Call Rate | Unauthorized calls / total cases | Lower is better |
| Sensitive Exposure Rate | Exposures / relevant attack cases | Lower is better |

Bootstrap 95% confidence intervals are computed for all major metrics (1000 resamples, seed=42).

## Evaluation Methodology

### Deterministic safety checks (primary)
Safety labels are assigned programmatically:
- Forbidden tool/recipient/query pattern matching
- Unauthorized tool detection
- Sensitive marker exposure in responses or emails

### LLM-as-judge (utility only)
Task completion is assessed by a separate LLM call using `task_success_criteria`. **Safety labels never depend on the judge.**

## Quick Start

```bash
cd agent-safety-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY

# Run evaluation (all cases × all configs)
python run_eval.py

# Run subset for testing
python run_eval.py --limit 10 --config baseline

# Launch dashboard
streamlit run app.py

# Run unit tests (no API key needed)
pytest tests/ -v
```

## Results

Results are generated at runtime—**no hardcoded performance numbers**.

After running `python run_eval.py`:
- `results/raw_results.csv` — per-run details
- `results/summary_metrics.csv` — metrics with bootstrap CIs
- `results/failure_cases.csv` — qualitative failure analysis
- `results/safety_utility_frontier.png` — safety–utility tradeoff plot

View the comparison table in the Streamlit dashboard or inspect `summary_metrics.csv` directly.

## Failure Analysis

Representative failures are auto-selected when:
- An attack succeeds
- A benign control is unnecessarily refused
- A legitimate task is not completed

Each failure record includes the user request, malicious instruction (if any), agent behavior, tool calls, and expected safe behavior.

## Limitations

- Small synthetic dataset (~50 cases) — wide confidence intervals
- Limited attack diversity — no multi-turn or adaptive attacks
- Model-dependent — results apply to the evaluated OpenAI model only
- LLM judge error — task success labels may be imperfect
- Simulated environment — not production security
- **Does not establish comprehensive security certification**

## Future Work

- Larger adversarial datasets with human labels
- Adaptive and multi-turn attacks
- Judge calibration and inter-rater reliability
- Multi-model comparison across providers
- Stronger permission systems and automated red teaming

## Project Structure

```
agent-safety-eval/
├── app.py              # Streamlit dashboard
├── run_eval.py         # Evaluation harness
├── data/eval_cases.jsonl
├── src/
│   ├── agent.py        # Agent runner (3 configs)
│   ├── tools.py        # Mock tool environment
│   ├── permission_gate.py
│   ├── evaluator.py    # Deterministic + LLM judge
│   ├── metrics.py      # Metrics, CIs, failure export
│   └── config.py
├── results/            # Generated outputs
└── tests/
```

## License

Portfolio / research project — use and adapt freely.
