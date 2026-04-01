import pytest
from httpx import ASGITransport, AsyncClient

from gigachat_openai_proxy.main import app, gc_client, settings
from gigachat_openai_proxy.settings import Settings


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
async def test_sequential_requests(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
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
