from pathlib import Path

import gigachat_openai_proxy.tls as tls_mod
from gigachat_openai_proxy.tls import existing_pem_path


def test_existing_pem_path_prefers_existing(tmp_path, monkeypatch):
    p = tmp_path / "certs" / "mincifry.pem"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tls_mod, "_repo_root", lambda: tmp_path)
    assert existing_pem_path() == str(p.resolve())

