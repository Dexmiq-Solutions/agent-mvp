"""Unit tests for AgentConfig loading, resolution, validation, and secret masking."""

import os
from unittest.mock import patch
import pytest

from agents.runtime.config import AgentConfig
from core.config import Settings
from exceptions.agent import AgentConfigurationError


def test_agent_config_defaults():
    """Verify AgentConfig initializes with safe default parameters."""
    config = AgentConfig()
    assert config.model == "gemini-2.5-flash"
    assert config.api_key is None
    assert config.base_url is None
    assert config.temperature == 0.0
    assert config.max_tokens is None
    assert config.timeout == 60.0
    assert config.max_retries == 2
    assert "helpful AI assistant" in config.system_prompt


def test_agent_config_repr_masks_secrets():
    """Verify __repr__ never leaks secret API keys."""
    config = AgentConfig(api_key="secret-openrouter-key-12345", model="test-model")
    repr_str = repr(config)
    assert "secret-openrouter-key-12345" not in repr_str
    assert "***" in repr_str
    assert "test-model" in repr_str


def test_agent_config_from_settings_openrouter_priority():
    """Verify OpenRouter-specific settings take precedence over generic LLM settings."""
    settings = Settings(
        OPENROUTER_API_KEY="openrouter-special-key",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1/",
        AGENT_MODEL="anthropic/claude-3.5-sonnet",
        AGENT_TEMPERATURE=0.7,
        AGENT_TIMEOUT=90.0,
        AGENT_MAX_RETRIES=4,
        OPENAI_API_KEY="openai-fallback-key",
        LLM_BASE_URL="https://api.openai.com/v1",
        LLM_MODEL="gpt-4o",
    )
    config = AgentConfig.from_settings(settings)
    assert config.api_key == "openrouter-special-key"
    assert config.base_url == "https://openrouter.ai/api/v1"  # trailing slash stripped
    assert config.model == "anthropic/claude-3.5-sonnet"
    assert config.temperature == 0.7
    assert config.timeout == 90.0
    assert config.max_retries == 4


def test_agent_config_fallback_to_llm_settings():
    """Verify fallback to generic LLM settings when OpenRouter settings are absent."""
    settings = Settings(
        OPENROUTER_API_KEY=None,
        OPENROUTER_BASE_URL=None,
        AGENT_MODEL=None,
        OPENAI_API_KEY="generic-openai-key",
        LLM_BASE_URL="https://custom.openai.endpoint/v1",
        LLM_MODEL="meta-llama/llama-3",
    )
    config = AgentConfig.from_settings(settings)
    assert config.api_key == "generic-openai-key"
    assert config.base_url == "https://custom.openai.endpoint/v1"
    assert config.model == "meta-llama/llama-3"


def test_agent_config_validation_requires_api_key():
    """Verify validation raises AgentConfigurationError when API key is missing."""
    config = AgentConfig(api_key=None, model="gpt-4o")
    with pytest.raises(AgentConfigurationError, match="Missing API key"):
        config.validate()

    empty_key_config = AgentConfig(api_key="   ", model="gpt-4o")
    with pytest.raises(AgentConfigurationError, match="Missing API key"):
        empty_key_config.validate()


def test_agent_config_validation_requires_model():
    """Verify validation raises AgentConfigurationError when model name is missing."""
    config = AgentConfig(api_key="valid-key", model="")
    with pytest.raises(AgentConfigurationError, match="model name must not be empty"):
        config.validate()


def test_agent_config_from_env_override():
    """Verify environment variable overrides take effect."""
    env = {
        "OPENROUTER_API_KEY": "env-or-key",
        "OPENROUTER_BASE_URL": "https://env.openrouter.ai/api/v1",
        "AGENT_MODEL": "env-agent-model",
    }
    with patch.dict(os.environ, env, clear=False):
        settings = Settings()
        config = AgentConfig.from_settings(settings)
        assert config.api_key == "env-or-key"
        assert config.base_url == "https://env.openrouter.ai/api/v1"
        assert config.model == "env-agent-model"
