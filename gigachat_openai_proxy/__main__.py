import argparse
import os

import uvicorn


def apply_serve_cli_to_environ(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="serve")
    p.add_argument("--debug", action="store_true", help="Эквивалент GIGACHAT_PROXY_DEBUG=true")
    args, _ = p.parse_known_args(argv)
    if args.debug:
        os.environ["GIGACHAT_PROXY_DEBUG"] = "true"


def main() -> None:
    apply_serve_cli_to_environ()
    uvicorn.run("gigachat_openai_proxy.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
