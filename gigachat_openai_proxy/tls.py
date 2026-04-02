import ssl
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_mincifry_pem_paths() -> list[Path]:
    # 1) cwd-relative for local runs
    # 2) repo-root-relative for cases when server cwd != repo root
    name = Path("certs") / "mincifry.pem"
    return [(Path.cwd() / name).resolve(), (_repo_root() / name).resolve()]


def default_ca_pem_paths() -> list[Path]:
    name = Path("certs") / "ca.pem"
    return [(Path.cwd() / name).resolve(), (_repo_root() / name).resolve()]


def existing_pem_path() -> str | None:
    for p in (default_ca_pem_paths() + default_mincifry_pem_paths()):
        try:
            if p.exists() and p.stat().st_size:
                return str(p)
        except OSError:
            pass
    return None


def verify_arg(verify_ssl: bool, ca_bundle: str | None) -> bool | ssl.SSLContext:
    if not verify_ssl:
        return False
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    p = existing_pem_path()
    if p:
        return ssl.create_default_context(cafile=p)
    return True

