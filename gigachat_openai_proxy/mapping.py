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
    c = ollama_json["message"]["content"]
    return openai_response(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": c},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        client_model,
    )


def openai_response(gc: dict, client_model: str) -> dict:
    ch0 = gc["choices"][0]
    msg = ch0["message"]
    u = gc.get("usage")
    if not u:
        u = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "id": gc.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": gc.get("created") or int(time.time()),
        "model": client_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": msg["role"], "content": msg["content"]},
                "finish_reason": ch0.get("finish_reason") or "stop",
            }
        ],
        "usage": u,
    }


def auth_header(key: str) -> str:
    k = key.strip()
    return k if k.lower().startswith("basic ") else f"Basic {k}"
