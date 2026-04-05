import json

from gigachat_openai_proxy.ollama_tools import (
    DEFAULT_TOOLS,
    execute_tool_call,
    patch,
    read_dir,
    read_file,
    run_executor,
    tools_for_ollama,
    write_file,
)


def test_read_file_pyproject():
    txt = read_file("pyproject.toml")
    assert "gigachat-openai-proxy" in txt


def test_read_dir_lists_package_files():
    names = json.loads(read_dir("gigachat_openai_proxy"))
    assert "main.py" in names


def test_read_file_dotdot_in_path_allowed():
    out = read_file("gigachat_openai_proxy/../pyproject.toml")
    assert "gigachat-openai-proxy" in out


def test_tools_for_ollama_defaults():
    assert tools_for_ollama([]) == DEFAULT_TOOLS
    assert tools_for_ollama(None) == DEFAULT_TOOLS


def test_execute_tool_call_dispatch():
    tc = {"function": {"name": "read_file", "arguments": {"path": "pyproject.toml"}}}
    out = execute_tool_call(tc)
    assert "gigachat-openai-proxy" in out


def test_run_executor_same_as_dispatch():
    out = run_executor("read_file", {"path": "pyproject.toml"})
    assert "gigachat-openai-proxy" in out


def test_write_file_roundtrip(tmp_path):
    p = tmp_path / "sub" / "t.txt"
    assert "ok" in write_file(str(p), "hello")
    assert p.read_text(encoding="utf-8") == "hello"


def test_execute_write_file(tmp_path):
    p = tmp_path / "w.txt"
    out = execute_tool_call(
        {"function": {"name": "write_file", "arguments": {"path": str(p), "content": "x"}}}
    )
    assert "ok" in out
    assert p.read_text() == "x"


def test_default_tools_includes_write_file():
    names = {t["function"]["name"] for t in DEFAULT_TOOLS}
    assert names == {"read_file", "read_dir", "write_file", "patch"}


def test_patch_one_occurrence(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("a\nb\na\n", encoding="utf-8")
    assert "not unique" in patch(str(p), "a", "Z", replace_all=False)
    assert p.read_text() == "a\nb\na\n"
    assert "ok" in patch(str(p), "b", "B", replace_all=False)
    assert p.read_text() == "a\nB\na\n"


def test_patch_replace_all(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x-x-", encoding="utf-8")
    assert "ok" in patch(str(p), "x", "y", replace_all=True)
    assert p.read_text() == "y-y-"


def test_patch_empty_old_rejected(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("z", encoding="utf-8")
    assert "non-empty" in patch(str(p), "", "x", replace_all=False)


def test_execute_patch(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("old", encoding="utf-8")
    out = execute_tool_call(
        {
            "function": {
                "name": "patch",
                "arguments": {"path": str(p), "old_string": "old", "new_string": "new"},
            }
        }
    )
    assert "ok" in out
    assert p.read_text() == "new"

