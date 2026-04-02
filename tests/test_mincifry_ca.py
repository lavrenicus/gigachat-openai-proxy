import io

import pytest

from gigachat_openai_proxy.mincifry_ca import ensure_ca_bundle


def test_ensure_ca_bundle_writes_pem(tmp_path, monkeypatch):
    pem = b"-----BEGIN CERTIFICATE-----\nQUJD\n-----END CERTIFICATE-----\n"

    def fake_urlopen(url, timeout=30):
        return io.BytesIO(pem)

    monkeypatch.setattr("gigachat_openai_proxy.mincifry_ca.urlopen", fake_urlopen)
    out = tmp_path / "x.pem"
    p = ensure_ca_bundle(str(out))
    assert out.read_text(encoding="utf-8") == pem.decode("utf-8")
    assert p == str(out.resolve())


def test_ensure_ca_bundle_converts_der(tmp_path, monkeypatch):
    der = b"\x01\x02\x03"

    def fake_urlopen(url, timeout=30):
        return io.BytesIO(der)

    monkeypatch.setattr("gigachat_openai_proxy.mincifry_ca.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "gigachat_openai_proxy.mincifry_ca.ssl.DER_cert_to_PEM_cert", lambda _: "PEM\n"
    )
    out = tmp_path / "x.pem"
    ensure_ca_bundle(str(out))
    assert out.read_text(encoding="utf-8") == "PEM\n"

