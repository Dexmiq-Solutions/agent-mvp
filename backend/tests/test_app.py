"""Tests for FastAPI application entrypoint and endpoints."""

from fastapi.testclient import TestClient

from app.main import app


def test_app_startup_and_health_check():
    """Verify application starts and health check returns expected status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app_name"] == "Agent MVP"
        assert data["environment"] == "development"


def test_root_endpoint():
    """Verify root endpoint returns welcome message."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Welcome to Agent MVP" in data["message"]


def test_cors_preflight_allowed_origin():
    """Verify CORS preflight OPTIONS request succeeds for allowed local origin."""
    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_get_allowed_origin_and_exposed_headers():
    """Verify CORS headers on actual GET request and check exposed headers."""
    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        expose_headers = response.headers.get("access-control-expose-headers", "")
        assert "retry-after" in expose_headers.lower()


def test_cors_disallowed_origin():
    """Verify CORS preflight fails for disallowed origins."""
    with TestClient(app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers


def test_cors_render_production_origin(monkeypatch):
    """Verify custom Render origin in CORS_ORIGINS is permitted."""
    monkeypatch.setenv("CORS_ORIGINS", "https://dexmiq-frontend.onrender.com")
    from core.config import get_settings
    get_settings.cache_clear()
    from app.main import create_application
    prod_app = create_application()

    with TestClient(prod_app) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "https://dexmiq-frontend.onrender.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://dexmiq-frontend.onrender.com"
        assert response.headers.get("access-control-allow-credentials") == "true"

    get_settings.cache_clear()

