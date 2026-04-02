import sys

import uvicorn

from gigachat_openai_proxy.app_config import AppConfig
from gigachat_openai_proxy.main import create_app

def _has(argv: list[str], flag: str) -> bool:
    return flag in argv


def main() -> None:
    args = sys.argv[1:]
    cfg = AppConfig(
        debug=_has(args, "--debug"),
        verify_ssl=not _has(args, "--no-verify-ssl"),
        use_mincifry_ca=_has(args, "--mincifry-ca"),
    )
    app = create_app(c=cfg)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
