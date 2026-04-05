import json

import pytest

from gigachat_openai_proxy.planner import (
    ACTION_PLANNER_FAILED,
    parse_plan,
    planner_messages,
    run_planner,
)
from gigachat_openai_proxy.router import filter_gigachat_messages


def test_parse_plan_final():
    p = parse_plan(json.dumps({"action": "final", "answer": "ok"}, ensure_ascii=False))
    assert p["action"] == "final" and p["answer"] == "ok"


def test_parse_plan_tool():
    p = parse_plan(json.dumps({"action": "tool", "tool": "read_file", "args": {"path": "x"}}))
    assert p["action"] == "tool" and p["tool"] == "read_file" and p["args"] == {"path": "x"}


def test_parse_plan_read_dir_short_form():
    p = parse_plan(json.dumps({"action": "read_dir", "args": {"path": "gigachat_openai_proxy"}}))
    assert p == {"action": "read_dir", "tool": None, "args": {"path": "gigachat_openai_proxy"}, "answer": ""}


def test_parse_plan_read_file_short_form():
    p = parse_plan(json.dumps({"action": "read_file", "args": {"path": "pyproject.toml"}}))
    assert p == {"action": "read_file", "tool": None, "args": {"path": "pyproject.toml"}, "answer": ""}


def test_parse_plan_strips_fence():
    t = '```json\n{"action":"final","answer":"hi"}\n```'
    assert parse_plan(t)["answer"] == "hi"


def test_parse_plan_strips_prefix_suffix():
    t = 'Думаю...\n{"action":"final","answer":"yes"}\nГотово.'
    assert parse_plan(t)["answer"] == "yes"


def test_parse_plan_first_object_when_two_json_blobs():
    t = '{"action":"read_file","args":{"path":"x"}}\n\n{"action":"final","answer":"y"}'
    assert parse_plan(t)["action"] == "read_file"


def test_parse_plan_invalid():
    with pytest.raises(ValueError):
        parse_plan("not json")


def test_planner_messages_prefixed_system():
    m = planner_messages([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert m[0]["role"] == "system"
    assert m[1:] == filter_gigachat_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )


class _BadPlannerGC:
    async def chat(self, body: dict) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "просто текст"}}]}


@pytest.mark.asyncio
async def test_run_planner_parse_fail_returns_delegation_action():
    p = await run_planner(_BadPlannerGC(), {}, "GigaChat:latest", dialog=[{"role": "user", "content": "x"}])
    assert p == {"action": ACTION_PLANNER_FAILED, "answer": "", "tool": None, "args": {}}
