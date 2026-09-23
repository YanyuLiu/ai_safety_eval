"""Mock tool environment for agent safety evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: str | None = None
    blocked: bool = False
    block_reason: str | None = None


@dataclass
class ToolTrace:
    requested: list[ToolCallRecord] = field(default_factory=list)
    executed: list[ToolCallRecord] = field(default_factory=list)
    blocked: list[ToolCallRecord] = field(default_factory=list)

    def record_request(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        record = ToolCallRecord(tool_name=tool_name, arguments=arguments)
        self.requested.append(record)
        return record

    def record_execution(self, record: ToolCallRecord, result: str) -> None:
        record.result = result
        self.executed.append(record)

    def record_block(self, record: ToolCallRecord, reason: str) -> None:
        record.blocked = True
        record.block_reason = reason
        record.result = f"Tool call blocked: {reason}"
        self.blocked.append(record)

    def to_serializable(self) -> dict[str, list[dict[str, Any]]]:
        def serialize(records: list[ToolCallRecord]) -> list[dict[str, Any]]:
            return [
                {
                    "tool_name": r.tool_name,
                    "arguments": r.arguments,
                    "result": r.result,
                    "blocked": r.blocked,
                    "block_reason": r.block_reason,
                }
                for r in records
            ]

        return {
            "requested": serialize(self.requested),
            "executed": serialize(self.executed),
            "blocked": serialize(self.blocked),
        }


class MockToolEnvironment:
    """In-memory mock environment with local tool implementations."""

    def __init__(self, environment: dict[str, Any]) -> None:
        self.files: dict[str, str] = dict(environment.get("files", {}))
        self.emails: list[dict[str, str]] = list(environment.get("emails", []))
        self.calendar: list[dict[str, str]] = list(environment.get("calendar", []))
        self.user_profile: dict[str, Any] = dict(environment.get("user_profile", {}))
        self.sent_emails: list[dict[str, str]] = []
        self.trace = ToolTrace()

    def read_file(self, file_name: str) -> str:
        if file_name not in self.files:
            return f"Error: File '{file_name}' not found."
        return self.files[file_name]

    def search_email(self, query: str) -> str:
        query_lower = query.lower()
        matches = []
        for email in self.emails:
            haystack = " ".join(
                [email.get("subject", ""), email.get("from", ""), email.get("body", "")]
            ).lower()
            if query_lower in haystack:
                matches.append(email)
        if not matches:
            return "No emails found matching the query."
        return json.dumps(matches, indent=2)

    def send_email(self, recipient: str, body: str) -> str:
        self.sent_emails.append({"recipient": recipient, "body": body})
        return f"Email sent successfully to {recipient}."

    def read_calendar(self) -> str:
        if not self.calendar:
            return "No calendar events found."
        return json.dumps(self.calendar, indent=2)

    def create_calendar_event(self, title: str, start: str, end: str) -> str:
        event = {"title": title, "start": start, "end": end}
        self.calendar.append(event)
        return f"Calendar event '{title}' created from {start} to {end}."

    def get_user_profile(self) -> str:
        if not self.user_profile:
            return "No user profile available."
        return json.dumps(self.user_profile, indent=2)

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        dispatch = {
            "read_file": lambda: self.read_file(arguments["file_name"]),
            "search_email": lambda: self.search_email(arguments["query"]),
            "send_email": lambda: self.send_email(arguments["recipient"], arguments["body"]),
            "read_calendar": lambda: self.read_calendar(),
            "create_calendar_event": lambda: self.create_calendar_event(
                arguments["title"], arguments["start"], arguments["end"]
            ),
            "get_user_profile": lambda: self.get_user_profile(),
        }
        if tool_name not in dispatch:
            return f"Error: Unknown tool '{tool_name}'."
        return dispatch[tool_name]()


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {"type": "string", "description": "Name of the file to read."}
                    },
                    "required": ["file_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_email",
                "description": "Search the user's email inbox for messages matching a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for email content."}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string", "description": "Email address of the recipient."},
                        "body": {"type": "string", "description": "Email body content."},
                    },
                    "required": ["recipient", "body"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_calendar",
                "description": "Read the user's calendar events.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                "description": "Create a new calendar event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title."},
                        "start": {"type": "string", "description": "Start time (ISO format)."},
                        "end": {"type": "string", "description": "End time (ISO format)."},
                    },
                    "required": ["title", "start", "end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_profile",
                "description": "Retrieve the user's profile information.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def load_eval_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load evaluation cases from a JSONL file."""
    from src.config import DATA_PATH

    file_path = path if path is not None else DATA_PATH
    cases: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases
