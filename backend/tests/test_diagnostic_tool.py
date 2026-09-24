"""Unit tests for the deterministic diagnostic test tool."""

from tools.diagnostic import echo_diagnostic_tool


def test_echo_diagnostic_tool_execution():
    """Verify echo_diagnostic_tool accepts input and returns deterministic output."""
    result = echo_diagnostic_tool.invoke({"message": "hello world"})
    assert result == "[DIAGNOSTIC_OK] Echo: hello world"


def test_echo_diagnostic_tool_empty_input():
    """Verify tool handles empty input cleanly."""
    result = echo_diagnostic_tool.invoke({"message": ""})
    assert result == "[DIAGNOSTIC_OK] Echo: "


def test_echo_diagnostic_tool_special_characters():
    """Verify tool handles special characters and symbols deterministically."""
    special_input = "!@#$%^&*()_+{}[]:;<>?,./"
    result = echo_diagnostic_tool.invoke({"message": special_input})
    assert result == f"[DIAGNOSTIC_OK] Echo: {special_input}"


def test_echo_diagnostic_tool_metadata():
    """Verify tool name, description, and schema definition."""
    assert echo_diagnostic_tool.name == "echo_diagnostic_tool"
    assert "deterministic diagnostic tool" in echo_diagnostic_tool.description.lower()
    schema = echo_diagnostic_tool.args_schema.model_json_schema()
    assert "message" in schema["properties"]
