from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from docqa.api import create_app


@pytest.fixture()
def client(pipeline, settings):
    with TestClient(create_app(settings, pipeline)) as c:
        yield c


def test_health_ready_stats(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] == "ready"
    assert client.get("/v1/stats").json()["chunks"] > 0


def test_search_returns_ranked_sources_and_request_id(client):
    r = client.post("/v1/search", json={"query": "api rate limit", "k": 3},
                    headers={"x-request-id": "abc"})
    assert r.status_code == 200 and r.headers["x-request-id"] == "abc"
    body = r.json()
    assert 0 < len(body["results"]) <= 3 and body["mode"] == "hybrid"


def test_ask_answers_and_cites(client):
    r = client.post("/v1/ask", json={"question": "How long are database backups retained?"})
    body = r.json()
    assert r.status_code == 200 and not body["answer"]["abstained"]
    assert body["answer"]["citations"] and body["sources"]


def test_ask_abstains_on_out_of_scope(client):
    r = client.post("/v1/ask", json={"question": "What is the weather in Perth tomorrow?"})
    assert r.json()["answer"]["abstained"] is True


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "   "}, {"query": "x", "k": 0},
                                  {"query": "x", "k": 99}, {"query": "x", "mode": "nope"},
                                  {"query": "x" * 1001}])
def test_validation_errors_are_422(client, body):
    assert client.post("/v1/search", json=body).status_code == 422


def test_api_key_enforced_when_configured(pipeline, settings):
    app = create_app(replace(settings, api_key="s3cret"), pipeline)
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200  # probes stay open
        assert c.post("/v1/search", json={"query": "x"}).status_code == 401
        assert c.post("/v1/search", json={"query": "x"}, headers={"x-api-key": "bad"}).status_code == 401
        ok = c.post("/v1/search", json={"query": "backups"}, headers={"x-api-key": "s3cret"})
        assert ok.status_code == 200


def test_not_ready_when_index_missing(settings, tmp_path):
    app = create_app(replace(settings, index_dir=tmp_path / "missing"))
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503
        assert c.post("/v1/search", json={"query": "x"}).status_code == 503
