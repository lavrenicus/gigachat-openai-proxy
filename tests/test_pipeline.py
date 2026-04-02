import pytest
from httpx import ASGITransport, AsyncClient

from gigachat_openai_proxy.app_config import AppConfig
from gigachat_openai_proxy.main import create_app
from gigachat_openai_proxy.settings import Settings

app = create_app(Settings(gigachat_authorization_key="k"), AppConfig(pipeline_generator_token="t"))


@pytest.mark.asyncio
async def test_pipeline_process_ok():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/pipeline/process",
                params={"timeout_sec": 1},
                json={"content": "  hello   world  ", "metadata": {"source": "g1"}, "timestamp": 1},
            )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["output"]["content"] == "hello world"
    assert data["output"]["metadata"]["source"] == "g1"
    assert data["output"]["timestamp"] == 1


@pytest.mark.asyncio
async def test_pipeline_ingest_requires_token_when_set(monkeypatch):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r1 = await ac.post("/v1/pipeline/ingest", json={"content": "x", "metadata": {}, "timestamp": 1})
            r2 = await ac.post(
                "/v1/pipeline/ingest",
                headers={"x-generator-token": "t"},
                json={"content": "x", "metadata": {}, "timestamp": 1},
            )
    assert r1.status_code == 401
    assert r2.status_code == 200
    assert "id" in r2.json()


@pytest.mark.asyncio
async def test_pipeline_error_does_not_block_next():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r_bad = await ac.post(
                "/v1/pipeline/process",
                params={"timeout_sec": 1},
                json={"content": "x", "metadata": {"cause_error": True}, "timestamp": 1},
            )
            r_ok = await ac.post(
                "/v1/pipeline/process",
                params={"timeout_sec": 1},
                json={"content": " ok ", "metadata": {}, "timestamp": 2},
            )
    assert r_bad.status_code == 200
    assert r_bad.json()["status"] == "error"
    assert r_ok.json()["status"] == "ok"
    assert r_ok.json()["output"]["content"] == "ok"

