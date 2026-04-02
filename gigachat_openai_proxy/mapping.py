import json
import time
import uuid


def upstream_body(req: dict, gigachat_model: str) -> dict:
    out: dict = {"model": gigachat_model, "messages": req["messages"]}
    if "temperature" in req and req["temperature"] is not None:
        out["temperature"] = req["temperature"]
    if "max_tokens" in req and req["max_tokens"] is not None:
        out["max_tokens"] = req["max_tokens"]
    return out


def openai_from_ollama(ollama_json: dict, client_model: str) -> dict:
    m = ollama_json.get("message") or {}
    content = str(m.get("content") or "")
    tool_calls = m.get("tool_calls") or []

    if tool_calls:
        calls = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or {}
            args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
            calls.append(
                {
                    "id": tc.get("id") or uuid.uuid4().hex,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        msg = {"role": "assistant", "content": "", "tool_calls": calls}
        finish_reason = "tool_calls"
    else:
        msg = {"role": "assistant", "content": content}
        finish_reason = "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": client_model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def openai_response(gc: dict, client_model: str) -> dict:
    ch0 = gc["choices"][0]
    msg = ch0["message"]
    u = gc.get("usage") or {}
    # Continue критично парсит строгий OpenAI-формат: режем лишние поля usage
    prompt_tokens = int(u.get("prompt_tokens") or 0)
    completion_tokens = int(u.get("completion_tokens") or 0)
    total_tokens = int(u.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        # Continue часто парсит/валит ответ, если `id` не похож на OpenAI `chatcmpl-*`
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(gc.get("created") or time.time()),
        "model": client_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(msg.get("content") or "")},
                "finish_reason": ch0.get("finish_reason") or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def auth_header(key: str) -> str:
    k = key.strip()
    return k if k.lower().startswith("basic ") else f"Basic {k}"
