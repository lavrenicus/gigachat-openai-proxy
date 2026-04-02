import os

os.environ.setdefault("GIGACHAT_AUTHORIZATION_KEY", "dGVzdDp0ZXN0")


def pytest_addoption(parser):
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="run full e2e tests (may spend tokens)",
    )
