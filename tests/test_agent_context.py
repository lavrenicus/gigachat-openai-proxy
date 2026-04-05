import json

from gigachat_openai_proxy.agent_context import (
    PLANNER_TAIL_MESSAGE_COUNT,
    compress_planner_state,
    count_recent_failed_patch_results,
    first_user_task_text,
    internal_critic_final_without_mutation,
    known_environment_message,
    record_tool_for_known_files,
    soft_resource_nudge_message,
    verify_final_against_edit_intent,
)
from gigachat_openai_proxy.planner import tool_result_message


def _tr(tool: str, args: dict, result: str) -> dict:
    return tool_result_message(tool, args, result)


def test_compress_dedupes_middle_tool_results():
    u = {"role": "user", "content": "task"}
    a, b = {"path": "x"}, {"path": "y"}
    msgs = [u, _tr("read_file", a, "v1"), _tr("read_dir", b, "[]"), _tr("read_file", a, "v2-final"), {"role": "user", "content": "t0"}, {"role": "user", "content": "t1"}]
    out = compress_planner_state(msgs, tail_n=2, known_files={})
    assert len(out) == 5
    bodies = [json.loads(m["content"].split(" ", 1)[1]) for m in out[1:] if m["content"].startswith("[tool_result]")]
    assert len(bodies) == 2
    assert bodies[1]["result"] == "v2-final"


def test_compress_preserves_tail_unchanged():
    u = {"role": "user", "content": "t"}
    tail = [{"role": "user", "content": f"tail{i}"} for i in range(PLANNER_TAIL_MESSAGE_COUNT)]
    mid = [_tr("read_file", {"path": "p"}, "a"), _tr("read_file", {"path": "p"}, "b")]
    msgs = [u, *mid, *tail]
    out = compress_planner_state(msgs, tail_n=PLANNER_TAIL_MESSAGE_COUNT, known_files={})
    assert out[-PLANNER_TAIL_MESSAGE_COUNT:] == tail


def test_known_environment_message_none_when_empty():
    assert known_environment_message({}) is None


def test_known_environment_message_lists_paths():
    m = known_environment_message({"a.txt": "read len=3"})
    assert m and m["role"] == "user" and "[Known Environment]" in m["content"] and "a.txt" in m["content"]


def test_soft_nudge_prefix():
    assert soft_resource_nudge_message()["content"].startswith("[planner_hint]")


def test_compress_inserts_snapshot_before_tail():
    u = {"role": "user", "content": "go"}
    tail = [{"role": "user", "content": "t0"}, {"role": "user", "content": "t1"}]
    mid = [_tr("read_file", {"path": "p"}, "body")]
    msgs = [u, *mid, *tail]
    known = {"/x": "read sha256:abc len=1"}
    out = compress_planner_state(msgs, tail_n=2, known_files=known)
    assert out[0] == u
    assert out[1]["content"].startswith("[tool_result]")
    assert "[Known Environment]" in out[2]["content"]
    assert out[-2:] == tail


def test_record_write_ignores_error_result():
    k: dict[str, str] = {}
    record_tool_for_known_files("write_file", {"path": "a.txt"}, "write_file error: boom", k)
    assert k == {}


def test_record_patch_only_prefix_ok():
    k = {}
    record_tool_for_known_files("patch", {"path": "a.txt"}, "patch error: x", k)
    assert k == {}
    record_tool_for_known_files("patch", {"path": "a.txt"}, "patch: ok: /a.txt", k)
    assert k["a.txt"] == "patched"


def test_verify_final_rejects_edit_without_mutation():
    plan = {"action": "final", "answer": "x"}
    assert not verify_final_against_edit_intent(plan, "добавь строку в f.txt", {})
    assert verify_final_against_edit_intent(plan, "добавь строку в f.txt", {"f.txt": "written"})
    assert verify_final_against_edit_intent(plan, "просто прочитай", {})


def test_first_user_task_skips_tool_result():
    st = [{"role": "user", "content": "[tool_result] {}"}, {"role": "user", "content": "реальная задача"}]
    assert first_user_task_text(st) == "реальная задача"


def test_count_recent_failed_patch_results():
    st = [
        {"role": "user", "content": "x"},
        tool_result_message("patch", {"path": "p"}, "patch: old_string not found"),
        tool_result_message("patch", {"path": "p"}, "patch: ok: p"),
    ]
    assert count_recent_failed_patch_results(st) == 1


def test_internal_critic_contains_prefix():
    m = internal_critic_final_without_mutation(
        draft="x" * 500 + "\n" * 10,
        known_files={"a": "read"},
        failed_patches=4,
        step=55,
        max_steps=72,
    )
    assert m["content"].startswith("[internal_critic]")
