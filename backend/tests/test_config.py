"""Tests for application configuration."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_default_settings():
    """Verify default settings load cleanly without any external credentials."""
    settings = Settings()
    assert settings.APP_NAME == "Agent MVP"
    assert settings.ENVIRONMENT == "development"
    assert settings.LOG_LEVEL == "INFO"
    assert settings.DEBUG is False
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000

    # Ensure all future integration credentials default to None or safe defaults
    assert settings.SUPABASE_URL is None
    assert settings.SUPABASE_KEY is None
    assert settings.SUPABASE_SERVICE_ROLE_KEY is None
    assert settings.SUPABASE_STORAGE_BUCKET == "documents"
    assert settings.supabase_storage_key is None
    assert settings.DATABASE_URL is None
    assert settings.QDRANT_URL is None
    assert settings.QDRANT_API_KEY is None
    assert settings.VOYAGE_API_KEY is None
    assert settings.OPENAI_API_KEY is None


def test_custom_environment_settings(monkeypatch):
    """Verify settings load overrides from environment variables."""
    monkeypatch.setenv("APP_NAME", "Custom Agent")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PORT", "9000")

    settings = Settings()
    assert settings.APP_NAME == "Custom Agent"
    assert settings.ENVIRONMENT == "production"
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.PORT == 9000


def test_supabase_storage_key_resolution(monkeypatch):
    """Verify supabase_storage_key prefers service role key over client key."""
    monkeypatch.setenv("SUPABASE_KEY", "anon_key")
    settings = Settings()
    assert settings.supabase_storage_key == "anon_key"

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service_role_secret")
    settings = Settings()
    assert settings.supabase_storage_key == "service_role_secret"



def test_invalid_log_level():
    """Verify invalid LOG_LEVEL raises a validation error."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(LOG_LEVEL="INVALID_LEVEL")
    assert "Invalid LOG_LEVEL" in str(exc_info.value)


def test_invalid_environment():
    """Verify invalid ENVIRONMENT raises a validation error."""
    with pytest.raises(ValidationError):
        Settings(ENVIRONMENT="invalid_env")


def test_get_settings_caching():
    """Verify get_settings returns the same cached instance."""
    settings_1 = get_settings()
    settings_2 = get_settings()
    assert settings_1 is settings_2
