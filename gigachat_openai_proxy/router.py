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
    # Роутим в Ollama только если в system-сообщении присутствует подсказка tool-протокола.
    needle = "tool_name"
    for m in messages:
        if m.get("role") == "system" and needle in _msg_text(m).lower():
            return True
    return False


def filter_gigachat_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") != "system"]
