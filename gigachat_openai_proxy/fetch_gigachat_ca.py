"""Докачка CA для GigaChat по официальной доке Сбера (gu-st.ru)."""

import sys
from pathlib import Path
from urllib.request import urlopen

# https://developers.sber.ru/docs/ru/gigachat/certificates
RTR_ROOT = "https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt"
RTR_SUB = "https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fetch(url: str) -> bytes:
    with urlopen(url, timeout=30) as r:
        return r.read()


def main(argv: list[str] | None = None) -> None:
    _ = argv if argv is not None else sys.argv[1:]
    d = _repo_root() / "certs"
    d.mkdir(parents=True, exist_ok=True)
    root_f = d / "russian_trusted_root_ca_pem.crt"
    sub_f = d / "russian_trusted_sub_ca_pem.crt"
    ca_f = d / "ca.pem"
    root_f.write_bytes(_fetch(RTR_ROOT))
    sub_f.write_bytes(_fetch(RTR_SUB))
    r = root_f.read_text(encoding="utf-8", errors="replace").strip()
    s = sub_f.read_text(encoding="utf-8", errors="replace").strip()
    ca_f.write_text(r + "\n" + s + "\n", encoding="utf-8")
    print(str(ca_f.resolve()))
