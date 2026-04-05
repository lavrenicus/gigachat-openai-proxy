import re
from pathlib import Path

_BLOCK = re.compile(r"```([^\n`]*?)\n([\s\S]*?)```", re.MULTILINE)


def inline_file_body_for_path(messages: list[dict], path: str) -> str | None:
    if not (path or "").strip():
        return None
    want = Path(path.replace("\\", "/")).name.casefold()
    for m in messages:
        if m.get("role") != "user":
            continue
        c = str(m.get("content") or "")
        if c.startswith("[tool_result]"):
            continue
        for mo in _BLOCK.finditer(c):
            info, body = mo.group(1).strip(), mo.group(2)
            names: list[str] = []
            if info:
                names.append(Path(info.replace("\\", "/")).name)
                for tok in info.split():
                    if "/" in tok or "\\" in tok:
                        names.append(Path(tok.replace("\\", "/")).name)
                    elif "." in tok:
                        names.append(tok)
            if any(n.casefold() == want for n in names if n):
                return body.rstrip("\n")
    return None
