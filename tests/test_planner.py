import json

import pytest

import gigachat_openai_proxy.planner as planner_mod
from gigachat_openai_proxy.planner import (
    ACTION_PLANNER_FAILED,
    parse_plan,
    planner_messages,
    run_planner,
    state_has_tool_result,
    tool_result_identity_key,
    tool_result_message,
)
from gigachat_openai_proxy.router import filter_gigachat_messages


def test_parse_plan_final():
    p = parse_plan(json.dumps({"action": "final", "answer": "ok"}, ensure_ascii=False))
    assert p["action"] == "final" and p["answer"] == "ok"


def test_parse_plan_tool():
    p = parse_plan(json.dumps({"action": "tool", "tool": "read_file", "args": {"path": "x"}}))
    assert p["action"] == "tool" and p["tool"] == "read_file" and p["args"] == {"path": "x"}


def test_parse_plan_write_file_short_form():
    p = parse_plan(
        json.dumps({"action": "write_file", "args": {"path": "a.txt", "content": "z"}}, ensure_ascii=False)
    )
    assert p == {"action": "write_file", "tool": None, "args": {"path": "a.txt", "content": "z"}, "answer": ""}


def test_parse_plan_tool_unknown_accepted():
    p = parse_plan(json.dumps({"action": "tool", "tool": "browser_navigate", "args": {}}))
    assert p == {"action": "tool", "tool": "browser_navigate", "args": {}, "answer": ""}


def test_parse_plan_unknown_action_short_form():
    p = parse_plan(json.dumps({"action": "made_up", "args": {"x": 1}}))
    assert p["action"] == "made_up" and p["args"] == {"x": 1}


def test_parse_plan_patch_short_form():
    p = parse_plan(
        json.dumps(
            {"action": "patch", "args": {"path": "a.txt", "old_string": "x", "new_string": "y"}},
            ensure_ascii=False,
        )
    )
    assert p["action"] == "patch" and p["args"]["old_string"] == "x"


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


def test_retry_user_contains_braced_json_example():
    assert '{"action":"read_dir","args":{"path":"..."}}' in planner_mod._RETRY_USER


def test_state_has_tool_result_detects_same_tool_args():
    st: list[dict] = []
    assert not state_has_tool_result(st, "read_dir", {"path": "C:\\projects"})
    st.append(tool_result_message("read_dir", {"path": "C:\\projects"}, '["a"]'))
    assert state_has_tool_result(st, "read_dir", {"path": "C:\\projects"})


def test_state_has_tool_result_same_args_different_key_order():
    st = [tool_result_message("read_dir", {"path": "x", "extra": 1}, "[]")]
    assert state_has_tool_result(st, "read_dir", {"extra": 1, "path": "x"})


def test_state_has_tool_result_different_path():
    st = [tool_result_message("read_dir", {"path": "a"}, "[]")]
    assert not state_has_tool_result(st, "read_dir", {"path": "b"})


def test_tool_result_identity_key_stable():
    m = tool_result_message("read_file", {"path": "z"}, "x")
    k1 = tool_result_identity_key(m)
    assert k1 and k1 == tool_result_identity_key(m)


@pytest.mark.asyncio
async def test_run_planner_parse_fail_returns_delegation_action():
    p = await run_planner(_BadPlannerGC(), {}, "GigaChat:latest", dialog=[{"role": "user", "content": "x"}])
    assert p == {"action": ACTION_PLANNER_FAILED, "answer": "", "tool": None, "args": {}}
