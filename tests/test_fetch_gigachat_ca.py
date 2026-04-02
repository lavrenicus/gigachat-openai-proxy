from pathlib import Path

from gigachat_openai_proxy import fetch_gigachat_ca


def test_fetch_gigachat_ca_writes_bundle(tmp_path, monkeypatch, capsys):
    def fake_fetch(url: str) -> bytes:
        if "root" in url:
            return b"ROOT\n"
        return b"SUB\n"

    monkeypatch.setattr(fetch_gigachat_ca, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(fetch_gigachat_ca, "_fetch", fake_fetch)
    fetch_gigachat_ca.main([])
    ca = (tmp_path / "certs" / "ca.pem").read_bytes()
    assert b"ROOT" in ca and b"SUB" in ca
    assert capsys.readouterr().out.strip().endswith("ca.pem")
