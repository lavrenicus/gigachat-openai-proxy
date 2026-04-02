from gigachat_openai_proxy.router import filter_gigachat_messages, use_ollama


def test_use_ollama_gigachat_plain_russian():
    assert not use_ollama([{"role": "user", "content": "Напиши функцию на Python"}])


def test_use_ollama_tool_read_file_path():
    assert use_ollama([{"role": "user", "content": "прочитай файл src/main.py"}])


def test_use_ollama_marker_read_file():
    assert use_ollama([{"role": "user", "content": "use read_file tool"}])


def test_use_ollama_system_tool():
    assert use_ollama([{"role": "system", "content": "You have access to tools"}])


def test_filter_drops_system():
    m = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    assert filter_gigachat_messages(m) == [{"role": "user", "content": "y"}]
