"""Tests for application configuration."""

import pytest
from pydantic import ValidationError

from core.config import Settings, get_settings


def test_default_settings():
    """Verify default settings load cleanly without any external credentials."""
    settings = Settings(_env_file=None)
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
    assert settings.COHERE_API_KEY is None
    assert settings.VOYAGE_API_KEY is None
    assert settings.EMBEDDING_PROVIDER == "cohere"
    assert settings.EMBEDDING_MODEL == "embed-v4.0"
    assert settings.OPENAI_API_KEY is None

    # Contextual Enrichment default
    assert settings.CONTEXTUAL_ENRICHMENT_ENABLED is False
    assert settings.CONTEXTUAL_ENRICHMENT_STRATEGY == "structured"

    # Reranking defaults
    assert settings.JINA_API_KEY is None
    assert settings.RERANKER_PROVIDER == "jina"
    assert settings.RERANKER_MODEL == "jina-reranker-v3.5"
    assert settings.RERANKER_TIMEOUT == 15.0


def test_custom_environment_settings(monkeypatch):
    """Verify settings load overrides from environment variables."""
    monkeypatch.setenv("APP_NAME", "Custom Agent")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_ENABLED", "true")
    monkeypatch.setenv("CONTEXTUAL_ENRICHMENT_STRATEGY", "custom")
    monkeypatch.setenv("RERANKER_PROVIDER", "jina")
    monkeypatch.setenv("RERANKER_MODEL", "jina-reranker-v3.5")
    monkeypatch.setenv("JINA_API_KEY", "test-jina-key")
    monkeypatch.setenv("RERANKER_TIMEOUT", "25.0")

    settings = Settings(_env_file=None)
    assert settings.APP_NAME == "Custom Agent"
    assert settings.ENVIRONMENT == "production"
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.PORT == 9000
    assert settings.CONTEXTUAL_ENRICHMENT_ENABLED is True
    assert settings.CONTEXTUAL_ENRICHMENT_STRATEGY == "custom"
    assert settings.RERANKER_PROVIDER == "jina"
    assert settings.RERANKER_MODEL == "jina-reranker-v3.5"
    assert settings.JINA_API_KEY == "test-jina-key"
    assert settings.RERANKER_TIMEOUT == 25.0


def test_supabase_storage_key_resolution(monkeypatch):
    """Verify supabase_storage_key prefers service role key over client key."""
    monkeypatch.setenv("SUPABASE_KEY", "anon_key")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.supabase_storage_key == "anon_key"

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service_role_secret")
    settings = Settings(_env_file=None)
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


def test_cors_origins_default_and_parsing(monkeypatch):
    """Verify CORS_ORIGINS defaults and parses comma-separated, JSON, and list formats."""
    # Default settings
    settings = Settings(_env_file=None)
    assert "http://localhost:5173" in settings.CORS_ORIGINS
    assert "http://127.0.0.1:5173" in settings.CORS_ORIGINS

    # Comma-separated string
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173,https://my-app.onrender.com")
    s2 = Settings(_env_file=None)
    assert s2.CORS_ORIGINS == ["http://localhost:5173", "https://my-app.onrender.com"]

    # JSON array string
    monkeypatch.setenv("CORS_ORIGINS", '["https://render-frontend.onrender.com"]')
    s3 = Settings(_env_file=None)
    assert s3.CORS_ORIGINS == ["https://render-frontend.onrender.com"]

    # Single string
    monkeypatch.setenv("CORS_ORIGINS", "https://single.onrender.com")
    s4 = Settings(_env_file=None)
    assert s4.CORS_ORIGINS == ["https://single.onrender.com"]

