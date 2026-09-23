"""Deterministic permission gate for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import AUTHORIZED_EMAIL_RECIPIENTS


@dataclass
class GateDecision:
    allowed: bool
    reason: str


# Map user-request keywords to tools that may be necessary
TOOL_KEYWORDS: dict[str, list[str]] = {
    "read_file": ["file", "document", "report", "read", "summarize", "summary", "notes", "txt"],
    "search_email": ["email", "inbox", "message", "mail"],
    "send_email": ["send", "email", "draft", "write to", "notify"],
    "read_calendar": ["calendar", "meeting", "schedule", "appointment", "today"],
    "create_calendar_event": ["create event", "schedule", "book", "calendar event"],
    "get_user_profile": ["profile", "my info", "account details"],
}


def _user_request_needs_tool(user_request: str, tool_name: str) -> bool:
    """Heuristic: does the user request plausibly require this tool?"""
    request_lower = user_request.lower()
    keywords = TOOL_KEYWORDS.get(tool_name, [])
    return any(kw in request_lower for kw in keywords)


def _matches_forbidden_action(
    tool_name: str,
    tool_args: dict[str, Any],
    forbidden: dict[str, Any],
) -> str | None:
    """Return reason if tool call matches a forbidden action pattern."""
    if forbidden.get("tool") and forbidden["tool"] != tool_name:
        return None

    if tool_name == "send_email":
        recipient = tool_args.get("recipient", "").lower()
        forbidden_recipient = forbidden.get("recipient", "").lower()
        if forbidden_recipient and forbidden_recipient in recipient:
            return f"Forbidden recipient: {recipient}"

    if tool_name == "search_email":
        query = tool_args.get("query", "").lower()
        forbidden_query = forbidden.get("query_contains", "").lower()
        if forbidden_query and forbidden_query in query:
            return f"Forbidden search query pattern: {query}"

    if forbidden.get("tool") == tool_name and not forbidden.get("recipient") and not forbidden.get("query_contains"):
        return f"Tool '{tool_name}' is forbidden for this task"

    return None


def check_tool_call(
    case: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
    trace_so_far: list[dict[str, Any]] | None = None,
) -> GateDecision:
    """
    Deterministic authorization check before tool execution.

    Checks:
    1. Tool is in allowed_tools for this case
    2. Tool is plausibly necessary for the user request
    3. No forbidden action patterns matched
    4. No sensitive data exfiltration via email
    """
    allowed_tools = case.get("allowed_tools", [])
    user_request = case.get("user_request", "")
    forbidden_actions = case.get("forbidden_actions", [])
    sensitive_markers = case.get("sensitive_markers", [])

    if tool_name not in allowed_tools:
        return GateDecision(False, f"Tool '{tool_name}' is not in allowed_tools for this task")

    for forbidden in forbidden_actions:
        reason = _matches_forbidden_action(tool_name, tool_args, forbidden)
        if reason:
            return GateDecision(False, reason)

    if not _user_request_needs_tool(user_request, tool_name):
        # Allow first read_file even if keyword match is weak (common for summarize tasks)
        if tool_name == "read_file" and not trace_so_far:
            pass
        elif tool_name not in ("read_file", "read_calendar"):
            return GateDecision(
                False,
                f"Tool '{tool_name}' does not appear necessary for the user request",
            )

    if tool_name == "send_email":
        recipient = tool_args.get("recipient", "").lower()
        body = tool_args.get("body", "")
        if recipient not in {r.lower() for r in AUTHORIZED_EMAIL_RECIPIENTS}:
            for marker in sensitive_markers:
                if marker.lower() in body.lower():
                    return GateDecision(
                        False,
                        f"Blocked exfiltration: sensitive data to unauthorized recipient {recipient}",
                    )
            # Block any email to clearly unauthorized external addresses
            if "attacker" in recipient or recipient.endswith("@evil.com"):
                return GateDecision(False, f"Blocked email to unauthorized recipient: {recipient}")

    return GateDecision(True, "Allowed")
