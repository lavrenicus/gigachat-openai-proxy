import ssl
from pathlib import Path
from urllib.request import urlopen


MINCIFRY_CA_URL = "https://ca.gisca.ru/repository/%D0%9C%D0%98%D0%9D%D0%A6%D0%98%D0%A4%D0%A0%D0%AB.cer"


def ensure_ca_bundle(path: str, url: str = MINCIFRY_CA_URL) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if p.exists() and p.stat().st_size:
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = urlopen(url, timeout=30).read()
    if b"-----BEGIN CERTIFICATE-----" in data:
        pem = data.decode("utf-8", errors="strict")
    else:
        pem = ssl.DER_cert_to_PEM_cert(data)
    p.write_text(pem, encoding="utf-8")
    return str(p)

