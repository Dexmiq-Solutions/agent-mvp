"""Unit tests for Agent model factory, provider abstraction, and credential handling."""

from unittest.mock import patch
import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agents.runtime.config import AgentConfig
from agents.runtime.model import create_agent_model
from exceptions.agent import AgentConfigurationError, AgentModelError


def test_create_agent_model_success():
    """Verify create_agent_model constructs a BaseChatModel instance with matching parameters."""
    config = AgentConfig(
        model="liquid/lfm-2.5-2.6b:free",
        api_key="test-api-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=500,
        timeout=45.0,
        max_retries=3,
    )
    model = create_agent_model(config)
    assert isinstance(model, BaseChatModel)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "liquid/lfm-2.5-2.6b:free"
    assert model.temperature == 0.2
    assert model.max_tokens == 500
    assert model.request_timeout == 45.0
    assert model.max_retries == 3
    assert str(model.openai_api_base).rstrip("/") == "https://openrouter.ai/api/v1"


def test_create_agent_model_missing_key_raises_configuration_error():
    """Verify create_agent_model raises AgentConfigurationError when API key is missing."""
    config = AgentConfig(
        model="test-model",
        api_key=None,
    )
    with pytest.raises(AgentConfigurationError, match="Missing API key"):
        create_agent_model(config)


def test_create_agent_model_secret_not_leaked_in_repr():
    """Verify that model object representation does not leak the raw secret API key."""
    secret_key = "super-secret-key-xyz-987"
    config = AgentConfig(
        model="test-model",
        api_key=secret_key,
        base_url="https://openrouter.ai/api/v1",
    )
    model = create_agent_model(config)
    assert secret_key not in repr(model)


def test_create_agent_model_wraps_unexpected_instantiation_errors():
    """Verify unexpected instantiation failures are caught and wrapped in AgentModelError."""
    config = AgentConfig(
        model="test-model",
        api_key="valid-key",
    )
    with patch("agents.runtime.model.ChatOpenAI", side_effect=RuntimeError("SDK failed")):
        with pytest.raises(AgentModelError, match="Failed to initialize Agent model"):
            create_agent_model(config)
