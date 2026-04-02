TOOL_MARKERS = (
    "TOOL_NAME",
    "read_file",
    "edit_existing_file",
    "run_terminal_command",
    "прочитай файл",
    "прочитай",
)


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            else:
                parts.append(str(p))
        return " ".join(parts)
    return str(c or "")


def use_ollama(messages: list[dict]) -> bool:
    if any(m.get("role") == "system" and "tool" in _msg_text(m).lower() for m in messages):
        return True
    return any(
        any(marker in _msg_text(m) for marker in TOOL_MARKERS) for m in messages
    )


def filter_gigachat_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") != "system"]
