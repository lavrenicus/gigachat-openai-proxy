import pytest

from gigachat_openai_proxy.__main__ import _has


def test_has_true():
    assert _has(["--debug"], "--debug")


def test_has_false():
    assert not _has([], "--debug")
