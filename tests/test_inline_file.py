from gigachat_openai_proxy.inline_file import inline_file_body_for_path


def test_inline_matches_fence_filename():
    msgs = [
        {
            "role": "user",
            "content": "This is the currently open file:\n\n```TODO.MD\n###TODO###\n```\n",
        }
    ]
    assert inline_file_body_for_path(msgs, "TODO.MD") == "###TODO###"
    assert inline_file_body_for_path(msgs, "./TODO.MD") == "###TODO###"


def test_inline_ignores_tool_result_messages():
    msgs = [
        {"role": "user", "content": "```X.MD\nbody\n```"},
        {"role": "user", "content": '[tool_result] {"tool":"read_file"}'},
    ]
    assert inline_file_body_for_path(msgs, "X.MD") == "body"


def test_inline_windows_path_basename():
    msgs = [{"role": "user", "content": "```TODO.MD\nline\n```"}]
    assert inline_file_body_for_path(msgs, r"C:\Users\u\TODO.MD") == "line"
