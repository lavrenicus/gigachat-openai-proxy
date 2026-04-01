import time

import httpx
import pytest

from gigachat_openai_proxy.client import GigachatClient
from gigachat_openai_proxy.settings import Settings


@pytest.mark.asyncio
async def test_token_cached_single_oauth():
    oauth_calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "ngw.devices.sberbank.ru":
            oauth_calls["n"] += 1
            return httpx.Response(
                200,
                json={"access_token": "tok", "expires_at": time.time() + 3600},
            )
        assert req.headers.get("authorization") == "Bearer tok"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "x"}}]},
        )

    s = Settings(gigachat_authorization_key="k", gigachat_oauth_url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        c = GigachatClient(s, http=http)
        await c.chat({"model": "GigaChat:latest", "messages": []})
        await c.chat({"model": "GigaChat:latest", "messages": []})
        await c.aclose()
    assert oauth_calls["n"] == 1
