import sys

from gigachat_openai_proxy.app_config import AppConfig
from gigachat_openai_proxy.mincifry_ca import ensure_ca_bundle
from gigachat_openai_proxy.tls import default_mincifry_pem_paths


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    path = args[0] if args else str(default_mincifry_pem_paths()[0])
    print(ensure_ca_bundle(path))
