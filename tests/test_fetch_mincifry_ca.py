import io

from gigachat_openai_proxy import fetch_mincifry_ca


def test_fetch_mincifry_ca_prints_path(tmp_path, monkeypatch, capsys):
    def fake_ensure(path, url=None):
        p = tmp_path / "mincifry.pem"
        p.write_text("x", encoding="utf-8")
        return str(p)

    monkeypatch.setattr(fetch_mincifry_ca, "ensure_ca_bundle", fake_ensure)
    fetch_mincifry_ca.main([str(tmp_path / "out.pem")])
    out = capsys.readouterr().out.strip()
    assert out.endswith("mincifry.pem")

