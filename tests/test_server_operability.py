import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from afterimage.server.app import app


def test_health_reports_no_model_loaded_when_cache_is_empty():
    client = TestClient(app)
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False
    assert payload["loaded_model"] is None
    assert "cuda_available" in payload


def test_version_endpoint_matches_package_version():
    from afterimage import __version__

    client = TestClient(app)
    assert client.get("/api/version").json() == {"version": __version__}


def test_stats_endpoint_404s_before_any_generation():
    client = TestClient(app)
    resp = client.get("/api/stats")
    assert resp.status_code == 404
