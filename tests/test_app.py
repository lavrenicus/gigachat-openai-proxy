import json
import logging

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from gigachat_openai_proxy.main import app, gc_client, ollama_http, settings, config
from gigachat_openai_proxy.app_config import AppConfig
from gigachat_openai_proxy.settings import Settings
from gigachat_openai_proxy.planner import PLANNER_SYSTEM


class FakeGCPlannerFailThenDelegate:
    """Первые вызовы — чат планировщика с битым ответом; затем обычный ответ основной модели."""

    def __init__(self) -> None:
        self.last: dict | None = None
        self.history: list[dict] = []

    async def chat(self, body: dict) -> dict:
        self.last = body
        self.history.append(body)
        first = ((body.get("messages") or [{}])[0] or {}).get("content") or ""
        if first.startswith("Ты планировщик"):
            return {"choices": [{"message": {"role": "assistant", "content": "не json"}}]}
        return {
            "choices": [{"message": {"role": "assistant", "content": "Сформулируйте иначе."}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    async def aclose(self) -> None:
        pass


def _ollama_unused_stub() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(500, text="ollama must not be called in this test")
        )
    )


class FakeGC:
    def __init__(self) -> None:
        self.last: dict | None = None
        self.history: list[dict] = []
        self.plan: dict | None = None
        self.plans: list[dict] | None = None

    async def chat(self, body: dict) -> dict:
        self.last = body
        self.history.append(body)
        if self.plans is not None:
            i = len(self.history) - 1
            plan = self.plans[i] if i < len(self.plans) else self.plans[-1]
        elif self.plan is not None:
            plan = self.plan
        else:
            plan = {"action": "final", "answer": "Ответ"}
        return {
            "id": "x",
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_gc():
    return FakeGC()


@pytest.mark.asyncio
async def test_planner_parse_fail_delegates_to_gigachat():
    s = Settings(gigachat_authorization_key="k", gigachat_model="GigaChat:latest")
    gc = FakeGCPlannerFailThenDelegate()
    app.dependency_overrides[gc_client] = lambda: gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"model": "gigachat", "messages": [{"role": "user", "content": "Привет"}]},
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "Сформулируйте иначе."
        assert len(gc.history) == 4
        assert "Ты планировщик" in gc.history[0]["messages"][0]["content"]
        assert gc.history[3]["messages"][0]["content"].startswith("Служебно")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_completions_roundtrip(fake_gc: FakeGC):
    s = Settings(
        gigachat_authorization_key="k",
        gigachat_model="GigaChat:latest",
    )
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
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
        assert data["usage"]["total_tokens"] == 0
        assert data["model"] == "gigachat"
        assert fake_gc.last and fake_gc.last["model"] == "GigaChat:latest"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_gigachat_filters_system(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
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
        assert fake_gc.last and fake_gc.last["messages"] == [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": "u"},
        ]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_routes_to_ollama(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gigachat",
                    "messages": [
                        {"role": "system", "content": "TOOL_NAME: read_file"},
                        {"role": "user", "content": "прочитай файл src/main.py"},
                    ],
                },
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "Ответ"
        assert r.json()["choices"][0]["finish_reason"] == "stop"
        assert fake_gc.last and fake_gc.last["messages"] == [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": "прочитай файл src/main.py"},
        ]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_plan_read_dir_action_executes_read_dir(caplog, fake_gc: FakeGC):
    _lg = logging.getLogger("gigachat_openai_proxy")
    _old, _lg.propagate = _lg.propagate, True
    caplog.set_level(logging.INFO)
    s = Settings(gigachat_authorization_key="k")
    fake_gc.plans = [
        {"action": "read_dir", "args": {"path": "gigachat_openai_proxy"}},
        {"action": "final", "answer": "listing-ok"},
    ]
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"model": "gigachat", "messages": [{"role": "user", "content": "листинг"}]},
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "listing-ok"
        assert any("Mapped action 'read_dir' → tool='read_dir'" in r.getMessage() for r in caplog.records)
        tr = fake_gc.history[1]["messages"][-1]["content"].split(" ", 1)[1]
        body1 = json.loads(tr)
        assert body1["tool"] == "read_dir" and body1["result"].startswith("[")
    finally:
        _lg.propagate = _old
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_custom_model_name_still_runs_planner(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    fake_gc.plans = [{"action": "final", "answer": "ok"}]
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"model": "my-openai-alias", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert r.status_code == 200
        assert r.json()["model"] == "my-openai-alias"
        assert str(fake_gc.history[0]["messages"][0].get("content", "")).startswith("Ты планировщик")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_read_file_inline_fence_skips_disk(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    fake_gc.plans = [
        {"action": "read_file", "args": {"path": "TODO.MD"}},
        {"action": "final", "answer": "done"},
    ]
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gigachat",
                    "messages": [
                        {
                            "role": "user",
                            "content": "open\n```TODO.MD\n###INLINE###\n```\n",
                        }
                    ],
                },
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "done"
        tr = fake_gc.history[1]["messages"][-1]["content"].split(" ", 1)[1]
        assert json.loads(tr)["result"] == "###INLINE###"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_plan_read_file_action_executes_read_file(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    fake_gc.plans = [
        {"action": "read_file", "args": {"path": "pyproject.toml"}},
        {"action": "final", "answer": "file-ok"},
    ]
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"model": "gigachat", "messages": [{"role": "user", "content": "файл"}]},
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "file-ok"
        tr = fake_gc.history[1]["messages"][-1]["content"].split(" ", 1)[1]
        body1 = json.loads(tr)
        assert body1["tool"] == "read_file" and "gigachat-openai-proxy" in body1["result"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_planner_loop_tool_then_final(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    fake_gc.plans = [
        {"action": "tool", "tool": "read_file", "args": {"path": "pyproject.toml"}},
        {"action": "final", "answer": "after-tool"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gigachat",
                    "messages": [
                        {"role": "system", "content": "TOOL_NAME: read_file"},
                        {"role": "user", "content": "Нужно читать файл"},
                    ],
                    "tools": tools,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["message"]["content"] == "after-tool"
        assert len(fake_gc.history) == 2
        msgs2 = fake_gc.history[1]["messages"]
        assert any(
            m.get("role") == "user" and str(m.get("content", "")).startswith("[tool_result]")
            for m in msgs2
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_tools_pass_through_still_planner_loop(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    fake_gc.plans = [
        {"action": "tool", "tool": "read_file", "args": {"path": "pyproject.toml"}},
        {"action": "final", "answer": "ok"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]

    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={"model": "gigachat", "messages": [{"role": "user", "content": "Нужно читать файл"}], "tools": tools},
            )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "ok"
        assert len(fake_gc.history) == 2
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sequential_requests(fake_gc: FakeGC):
    s = Settings(gigachat_authorization_key="k")
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
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


@pytest.mark.asyncio
async def test_stream_true_returns_sse(fake_gc: FakeGC):
    s = Settings(
        gigachat_authorization_key="k",
        gigachat_model="GigaChat:latest",
    )
    app.dependency_overrides[gc_client] = lambda: fake_gc
    app.dependency_overrides[settings] = lambda: s
    app.dependency_overrides[config] = lambda: AppConfig()
    app.dependency_overrides[ollama_http] = _ollama_unused_stub
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "gigachat",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "stream": True,
                },
            )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        assert "[DONE]" in r.text
        assert "chat.completion.chunk" in r.text
        assert "Ответ" in r.text
    finally:
        app.dependency_overrides.clear()
