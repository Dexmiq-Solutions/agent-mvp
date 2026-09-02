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
