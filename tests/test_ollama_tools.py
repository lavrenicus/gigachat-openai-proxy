import json

from gigachat_openai_proxy.ollama_tools import (
    DEFAULT_TOOLS,
    execute_tool_call,
    read_dir,
    read_file,
    run_executor,
    tools_for_ollama,
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

