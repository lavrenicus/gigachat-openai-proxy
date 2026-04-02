from gigachat_openai_proxy.mapping import auth_header, openai_from_ollama, openai_response, upstream_body


def test_upstream_maps_model_and_optional_fields():
    b = upstream_body(
        {"messages": [{"role": "user", "content": "h"}], "temperature": 0.5, "max_tokens": 10},
        "GigaChat:latest",
    )
    assert b == {
        "model": "GigaChat:latest",
        "messages": [{"role": "user", "content": "h"}],
        "temperature": 0.5,
        "max_tokens": 10,
    }


def test_upstream_skips_none_optional():
    b = upstream_body({"messages": [], "temperature": None}, "X")
    assert "temperature" not in b and "max_tokens" not in b


def test_openai_response_usage_fallback():
    r = openai_response(
        {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "length"}
            ]
        },
        "gigachat",
    )
    assert r["object"] == "chat.completion"
    assert r["model"] == "gigachat"
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["total_tokens"] == 0


def test_openai_response_passes_usage():
    u = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    r = openai_response(
        {"choices": [{"message": {"role": "assistant", "content": ""}}], "usage": u},
        "gigachat",
    )
    assert r["usage"] == u


def test_openai_from_ollama():
    r = openai_from_ollama({"message": {"role": "assistant", "content": "hi"}}, "gigachat")
    assert r["choices"][0]["message"]["content"] == "hi"
    assert r["object"] == "chat.completion"


def test_auth_header_adds_basic():
    assert auth_header("abc") == "Basic abc"


def test_auth_header_preserves_basic():
    assert auth_header("Basic abc") == "Basic abc"
