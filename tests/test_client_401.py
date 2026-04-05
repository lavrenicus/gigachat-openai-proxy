import time

import httpx
import pytest

from gigachat_openai_proxy.client import GigachatClient
from gigachat_openai_proxy.settings import Settings


@pytest.mark.asyncio
async def test_gigachat_chat_refetches_token_on_401():
    s = Settings(gigachat_authorization_key="a2E=")
    chat_calls = [0]

    def dispatch(req: httpx.Request) -> httpx.Response:
        if "/oauth" in str(req.url):
            return httpx.Response(
                200,
                json={"access_token": "tok", "expires_at": time.time() + 7200},
            )
        chat_calls[0] += 1
        if chat_calls[0] == 1:
            return httpx.Response(401, json={"message": "Token has expired"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    gc = GigachatClient(s, http=http, verify=False)
    try:
        out = await gc.chat({"model": "GigaChat:latest", "messages": [{"role": "user", "content": "x"}]})
        assert out["choices"][0]["message"]["content"] == "ok"
        assert chat_calls[0] == 2
    finally:
        await gc.aclose()
        await http.aclose()
