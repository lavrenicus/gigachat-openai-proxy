import json


def _shorten(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[truncated]"


def _sanitize_messages(messages: list[dict], content_limit: int) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        mm = dict(m)
        c = mm.get("content")
        if isinstance(c, str):
            mm["content"] = _shorten(c, content_limit)
        elif c is not None:
            mm["content"] = _shorten(str(c), content_limit)
        out.append(mm)
    return out


def pretty(obj: object, *, content_limit: int = 800, max_chars: int = 6000) -> str:
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        o = dict(obj)
        o["messages"] = _sanitize_messages(obj["messages"], content_limit)
    else:
        o = obj
    s = json.dumps(o, ensure_ascii=False, default=str, indent=2)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"


def pretty_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def pretty_openai_chat_response(resp: dict, *, content_limit: int = 2000) -> str:
    o = dict(resp)
    ch = o.get("choices")
    if isinstance(ch, list) and ch:
        c0 = dict(ch[0])
        msg = c0.get("message")
        if isinstance(msg, dict):
            c0["message"] = dict(msg)
            content = c0["message"].get("content")
            if not isinstance(content, str):
                content = str(content or "")
            c0["message"]["content"] = _shorten(content, content_limit)
        o["choices"] = [c0] + ch[1:]
    return json.dumps(o, ensure_ascii=False, default=str, indent=2)

