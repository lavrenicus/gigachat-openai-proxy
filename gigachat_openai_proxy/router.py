def filter_gigachat_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") != "system"]
