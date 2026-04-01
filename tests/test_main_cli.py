import os

import pytest

from gigachat_openai_proxy.__main__ import apply_serve_cli_to_environ


@pytest.fixture
def clean_debug_env():
    k = "GIGACHAT_PROXY_DEBUG"
    old = os.environ.pop(k, None)
    yield
    if old is not None:
        os.environ[k] = old
    else:
        os.environ.pop(k, None)


def test_debug_flag_sets_env(clean_debug_env):
    apply_serve_cli_to_environ(["--debug"])
    assert os.environ["GIGACHAT_PROXY_DEBUG"] == "true"


def test_no_flag_leaves_env(clean_debug_env):
    apply_serve_cli_to_environ([])
    assert "GIGACHAT_PROXY_DEBUG" not in os.environ
