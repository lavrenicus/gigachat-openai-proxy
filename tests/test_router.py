from gigachat_openai_proxy.router import filter_gigachat_messages


def test_filter_drops_system():
    m = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    assert filter_gigachat_messages(m) == [{"role": "user", "content": "y"}]
