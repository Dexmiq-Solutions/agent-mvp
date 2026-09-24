"""Deterministic diagnostic test tool for agent runtime validation."""

from langchain_core.tools import tool


@tool
def echo_diagnostic_tool(message: str) -> str:
    """A deterministic diagnostic tool that echoes back the provided input message.

    Used solely for runtime smoke testing and verifying agent tool execution paths.
    Has no external dependencies, no side effects, and produces predictable output.

    Args:
        message: Input text to echo back.

    Returns:
        A deterministic formatted string confirming tool receipt and execution.
    """
    return f"[DIAGNOSTIC_OK] Echo: {message}"
