from gigachat_openai_proxy.router import filter_gigachat_messages, use_ollama


def test_use_ollama_gigachat_plain_russian():
    assert not use_ollama([{"role": "user", "content": "Напиши функцию на Python"}])


def test_use_ollama_tool_read_file_path():
    assert use_ollama(
        [
            {"role": "system", "content": "TOOL_NAME: read_file"},
            {"role": "user", "content": "прочитай файл src/main.py"},
        ]
    )


def test_use_ollama_marker_read_file():
    assert use_ollama(
        [
            {"role": "system", "content": "TOOL_NAME: read_file"},
            {"role": "user", "content": "use read_file tool"},
        ]
    )


def test_use_ollama_system_tool():
    # system-секция обычно содержит шаблон tool-протокола для Continue,
    # и по ней нельзя переключать роутинг на Ollama.
    assert not use_ollama([{"role": "system", "content": "You have access to tools"}])


def test_use_ollama_when_user_requests_tools_russian():
    msgs = [
        {"role": "system", "content": "TOOL_NAME: read_file"},
        {"role": "user", "content": "Используя инструменты изучи этот проект"},
    ]
    assert use_ollama(msgs)


def test_use_ollama_not_when_user_greeting():
    msgs = [
        {"role": "system", "content": "You have access to tools"},
        {"role": "user", "content": "Привет"},
    ]
    assert not use_ollama(msgs)


def test_filter_drops_system():
    m = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    assert filter_gigachat_messages(m) == [{"role": "user", "content": "y"}]
