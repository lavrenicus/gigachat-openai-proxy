from gigachat_openai_proxy.mapping import (
    auth_header,
    openai_from_ollama,
    openai_from_text,
    openai_response,
    planner_completion_max_tokens,
    upstream_body,
)
import json


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


def test_planner_completion_max_tokens_floor():
    assert planner_completion_max_tokens(30) == 512
    assert planner_completion_max_tokens(None) == 2048
    assert planner_completion_max_tokens(4096) == 4096


def test_upstream_planner_sets_min_max_tokens():
    b = upstream_body(
        {"messages": [], "max_tokens": 30, "temperature": 0.1},
        "GigaChat:latest",
        planner=True,
    )
    assert b["max_tokens"] == 512
    assert b["temperature"] == 0.1


def test_upstream_planner_uses_client_when_above_min():
    b = upstream_body({"messages": [], "max_tokens": 900}, "M", planner=True)
    assert b["max_tokens"] == 900


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
    u = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "extra": 9}
    r = openai_response(
        {"choices": [{"message": {"role": "assistant", "content": ""}}], "usage": u},
        "gigachat",
    )
    assert r["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_openai_from_text():
    r = openai_from_text("x", "gigachat")
    assert r["choices"][0]["message"]["content"] == "x"
    assert r["usage"]["total_tokens"] == 0


def test_openai_from_ollama():
    r = openai_from_ollama({"message": {"role": "assistant", "content": "hi"}}, "gigachat")
    assert r["choices"][0]["message"]["content"] == "hi"
    assert r["object"] == "chat.completion"


def test_openai_from_ollama_tool_calls():
    r = openai_from_ollama(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "src/main.py"}}}
                ],
            }
        },
        "gigachat",
    )
    msg = r["choices"][0]["message"]
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"
    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "src/main.py"}


def test_auth_header_adds_basic():
    assert auth_header("abc") == "Basic abc"


def test_auth_header_preserves_basic():
    assert auth_header("Basic abc") == "Basic abc"
