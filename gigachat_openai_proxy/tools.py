from collections.abc import Callable

from gigachat_openai_proxy.ollama_tools import run_executor

_TOOL = run_executor

TOOL_REGISTRY: dict[str, Callable[[dict], str]] = {
    "read_file": lambda a: _TOOL("read_file", a),
    "read_dir": lambda a: _TOOL("read_dir", a),
    "write_file": lambda a: _TOOL("write_file", a),
    "patch": lambda a: _TOOL("patch", a),
}
