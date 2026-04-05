import json
from pathlib import Path
from typing import Any

from gigachat_openai_proxy.mapping import openai_from_ollama
from gigachat_openai_proxy.ollama_client import ollama_chat

DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file by filesystem path",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string", "description": "path"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_dir",
            "description": "List directory entries by filesystem path",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string", "description": "path"}},
            },
        },
    },
]


def tools_for_ollama(req_tools: list | None) -> list[dict[str, Any]]:
    if isinstance(req_tools, list) and req_tools:
        return req_tools
    return DEFAULT_TOOLS


def _resolved_path(path: str) -> Path:
    p = (path or "").strip()
    if not p:
        raise ValueError("empty path")
    return Path(p).expanduser().resolve()


def read_file(path: str) -> str:
    try:
        target = _resolved_path(path)
        if not target.exists() or not target.is_file():
            return f"read_file: not found: {path}"
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"read_file error: {e}"


def read_dir(path: str) -> str:
    try:
        target = _resolved_path(path)
        if not target.exists() or not target.is_dir():
            return f"read_dir: not a directory: {path}"
        names = sorted([p.name for p in target.iterdir()])
        return json.dumps(names, ensure_ascii=False)
    except Exception as e:
        return f"read_dir error: {e}"


def _parse_tool_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else {"value": v}
        except Exception:
            return {"value": args}
    return {"value": str(args)}


def run_executor(tool: str, args: dict[str, Any]) -> str:
    return execute_tool_call({"function": {"name": tool, "arguments": args}})


def execute_tool_call(tool_call: dict[str, Any]) -> str:
    fn = tool_call.get("function") or {}
    name = str(fn.get("name") or "")
    args = _parse_tool_args(fn.get("arguments"))

    if name == "read_file":
        return read_file(str(args.get("path") or ""))
    if name == "read_dir":
        return read_dir(str(args.get("path") or ""))
    return f"unknown tool: {name}"


async def ollama_tool_loop(
    http,
    s,
    *,
    initial_messages: list[dict],
    req_dump: dict,
    client_model: str,
    debug: bool = False,
    max_iters: int = 8,
) -> dict:
    messages = list(initial_messages)
    for _ in range(max_iters):
        raw_o = await ollama_chat(http, s, messages, req_dump, debug=debug)
        msg = raw_o.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return openai_from_ollama(raw_o, client_model)

        # Add assistant tool-call message for Ollama agent loop.
        messages.append(
            {
                "role": "assistant",
                "content": str(msg.get("content") or ""),
                "tool_calls": tool_calls,
            }
        )
        for tc in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_name": (tc.get("function") or {}).get("name") or "",
                    "content": execute_tool_call(tc),
                }
            )

    # Fallback: max iters reached.
    return openai_from_ollama({"message": {"role": "assistant", "content": ""}}, client_model)

