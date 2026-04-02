import json

import httpx
import pytest


def _json_from_text(s: str) -> dict:
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError(f"no json object in: {s[:200]!r}")
    return json.loads(s[i : j + 1])


@pytest.mark.asyncio
async def test_token_spend_pipeline_user_local_tools_llm_user(request: pytest.FixtureRequest):
    if not request.config.getoption("--full"):
        pytest.skip("run with --full (spends tokens + requires ollama)")
    base = "http://127.0.0.1:8000"
    async with httpx.AsyncClient(base_url=base, timeout=120) as ac:
        r_gen = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "local-tools",
                "messages": [
                    {"role": "system", "content": "TOOL_NAME: read_file"},
                    {
                        "role": "user",
                        "content": (
                            "Сгенерируй черновик ответа и верни СТРОГО JSON без пояснений: "
                            '{"content":"...","metadata":{"source":"generator1","type":"text","timestamp":1680358300},"timestamp":1680358300}'
                        ),
                    },
                ],
            },
        )
        assert r_gen.status_code == 200
        gen_txt = r_gen.json()["choices"][0]["message"]["content"]
        env = _json_from_text(gen_txt)
        assert isinstance(env.get("content"), str) and env["content"].strip()
        assert isinstance(env.get("metadata"), dict)

        r_llm = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "gigachat",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Ты процессор. Улучши/уточни ответ пользователя по этому черновику. "
                            "Убери мусор, дополни, но ответ должен быть коротким.\n\n"
                            f"Черновик(JSON): {json.dumps(env, ensure_ascii=False)}"
                        ),
                    }
                ],
            },
        )
    if r_llm.status_code != 200:
        b = r_llm.text
        if "CERTIFICATE_VERIFY_FAILED" in b or "self-signed certificate" in b:
            b += (
                "\n\nTLS для GigaChat: по доке Сбера нужны корни НУЦ (gu-st.ru). "
                "poetry run fetch-gigachat-ca — скачает root+sub в certs/ и соберёт certs/ca.pem; "
                "перезапусти прокси. См. https://developers.sber.ru/docs/ru/gigachat/certificates"
            )
        raise AssertionError(f"llm status={r_llm.status_code} body={b}")
    final_txt = r_llm.json()["choices"][0]["message"]["content"]
    assert isinstance(final_txt, str) and final_txt.strip()
