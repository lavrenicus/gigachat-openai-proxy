import os
import sys

import uvicorn


def apply_serve_cli_to_environ(argv: list[str] | None = None) -> None:
    # Точное вхождение токена: обходит кривой sys.argv у poetry run <script> без poetry install
    args = sys.argv[1:] if argv is None else argv
    if "--debug" in args:
        os.environ["GIGACHAT_PROXY_DEBUG"] = "true"


def main() -> None:
    apply_serve_cli_to_environ()
    uvicorn.run("gigachat_openai_proxy.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
