import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from gigachat_openai_proxy.main import app, gc_client, ollama_http, settings
from gigachat_openai_proxy.settings import Settings


def _ollama_unused_stub() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(500, text="ollama must not be called in this test")
        )
    )


class FakeGC:
    def __init__(self) -> None:
        self.last: dict | None = None

    async def chat(self, body: dict) -> dict:
        self.last = body
        return {
            "id": "x",
            "choices": [{"message": {"role": "assistant", "content": "Ответ"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_gc():
    return FakeGC()


@pytest.mark.asyncio
async def test_chat_completions_roundtrip(fake_gc: FakeGC):
    s = Settings(
        gigachat_authorization_key="k",
        gigachat_model="GigaChat:latest",
    )
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"model": "gigachat", "messages": [{"role": "user", "content": "Привет"}]},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"] == "Ответ"
        assert data["usage"]["total_tokens"] == 8
        assert data["model"] == "gigachat"
        assert fake_gc.last and fake_gc.last["model"] == "GigaChat:latest"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_gigachat_filters_system(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gigachat",
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "u"},
                    ],
                },
            )
        assert fake_gc.last and fake_gc.last["messages"] == [{"role": "user", "content": "u"}]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_routes_to_ollama(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")

    def handler(req):
        if req.url.path.endswith("/api/chat"):
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "ollama-ok"}})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as oh:
        app.dependency_overrides[gc_client] = lambda: fake_gc
        app.dependency_overrides[settings] = lambda: s
        app.dependency_overrides[ollama_http] = lambda: oh
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://t") as ac:
                r = await ac.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gigachat",
                        "messages": [{"role": "user", "content": "прочитай файл src/main.py"}],
                    },
                )
            assert r.status_code == 200
            assert r.json()["choices"][0]["message"]["content"] == "ollama-ok"
            assert fake_gc.last is None
        finally:
            app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sequential_requests(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            for _ in range(3):
                r = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "gigachat", "messages": [{"role": "user", "content": "x"}]},
                )
                assert r.status_code == 200
    finally:
        app.dependency_overrides.clear()
