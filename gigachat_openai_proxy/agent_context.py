import hashlib
import json

from gigachat_openai_proxy.planner import TOOL_RESULT_PREFIX, tool_result_identity_key

PLANNER_TAIL_MESSAGE_COUNT = 8
SOFT_STEP_LIMIT_BEFORE_NUDGE = 40

_EDIT_INTENT_WORDS = (
    "создай",
    "измени",
    "добавь",
    "удали",
    "пропатчи",
    "запиши",
    "обнови",
    "исправь",
    "перепиши",
    "внеси",
    "удалить",
    "создать",
    "create",
    "change",
    "add",
    "delete",
    "remove",
    "patch",
    "write",
    "edit",
    "fix",
    "update",
    "replace",
)

_SERVICE_PREFIXES = (
    "[tool_result]",
    "[tool_error]",
    "[planner_hint]",
    "[Known Environment]",
    "[internal_critic]",
)


def first_user_task_text(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") != "user":
            continue
        c = str(m.get("content") or "")
        if any(c.startswith(p) for p in _SERVICE_PREFIXES):
            continue
        return c
    return ""


def task_implies_disk_edit(task: str) -> bool:
    t = task.lower()
    return any(w in t for w in _EDIT_INTENT_WORDS)


def known_has_successful_mutation(known_files: dict[str, str]) -> bool:
    return any("written" in v or "patched" in v for v in known_files.values())


def verify_final_against_edit_intent(plan: dict, task_text: str, known_files: dict[str, str]) -> bool:
    if plan.get("action") != "final":
        return True
    if not task_implies_disk_edit(task_text):
        return True
    return known_has_successful_mutation(known_files)


def record_tool_for_known_files(tool: str, args: dict, res: str, known: dict[str, str]) -> None:
    r = str(res)
    lo = r.lower()
    if tool == "read_file":
        p = str(args.get("path") or "")
        if not p or "not found" in lo:
            return
        if lo.startswith("read_file error"):
            return
        h = hashlib.sha256(r.encode("utf-8", errors="replace")).hexdigest()[:12]
        known[p] = f"read sha256:{h} len={len(r)}"
    elif tool == "write_file":
        p = str(args.get("path") or "")
        if p and r.lstrip().startswith("write_file: ok"):
            known[p] = "written"
    elif tool == "patch":
        p = str(args.get("path") or "")
        if p and r.lstrip().startswith("patch: ok"):
            known[p] = "patched"


def count_recent_failed_patch_results(messages: list[dict], *, tail: int = 40) -> int:
    n, pref = 0, TOOL_RESULT_PREFIX
    for m in messages[-tail:]:
        c = str(m.get("content") or "")
        if not c.startswith(pref):
            continue
        try:
            body = json.loads(c[len(pref) :])
        except json.JSONDecodeError:
            continue
        if body.get("tool") != "patch":
            continue
        res = str(body.get("result") or "")
        if not res.lstrip().startswith("patch: ok"):
            n += 1
    return n


def internal_critic_final_without_mutation(
    *,
    draft: str,
    known_files: dict[str, str],
    failed_patches: int,
    step: int,
    max_steps: int,
) -> dict:
    parts = [
        "[internal_critic] Стоп: выбран action=final, но запрос требует изменить файлы на диске, "
        "а в Known Environment нет успешного write_file или patch.",
        "Проверь [Known Environment] и [tool_result], выясни почему запись не прошла; исправь план (строгий JSON) и выполни инструмент, затем снова final.",
    ]
    if known_files:
        brief = "; ".join(f"{k}={v}" for k, v in sorted(known_files.items())[:24])
        parts.append(f"Known Environment сейчас: {brief}")
    if failed_patches >= 3:
        parts.append(
            "Много неуспешных patch: отвечай одним JSON без markdown вокруг; при необходимости используй write_file с полным content."
        )
    if step >= 50 or step >= max_steps - 22:
        parts.append(
            "Шагов осталось мало: не обходи каталоги зря — сделай write_file/patch по целевому пути или честно опиши блокер."
        )
    if len(draft) > 400 and ("```" in draft or draft.count("\n") > 8):
        parts.append(
            "Похоже, ответ похож на содержимое файла в чате, но на диск ничего не записано — вызови инструмент записи."
        )
    return {"role": "user", "content": "\n".join(parts)}


def soft_resource_nudge_message() -> dict:
    return {
        "role": "user",
        "content": (
            "[planner_hint] Внимание: ресурсы почти на исходе. Если задача почти готова — используй action=final. "
            "Не начинай новых масштабных обходов каталогов."
        ),
    }


def _dedupe_tool_results_keep_last(messages: list[dict]) -> list[dict]:
    last: dict[str, int] = {}
    for i, m in enumerate(messages):
        k = tool_result_identity_key(m)
        if k is not None:
            last[k] = i
    return [m for i, m in enumerate(messages) if (k := tool_result_identity_key(m)) is None or last[k] == i]


def known_environment_message(known_files: dict[str, str]) -> dict | None:
    if not known_files:
        return None
    lines = [f"  {p}: {info}" for p, info in sorted(known_files.items())]
    return {
        "role": "user",
        "content": "[Known Environment]\n" + "Files on record:\n" + "\n".join(lines),
    }


def compress_planner_state(
    messages: list[dict], *, tail_n: int = PLANNER_TAIL_MESSAGE_COUNT, known_files: dict[str, str]
) -> list[dict]:
    if not messages:
        return messages
    fi = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    if fi is None:
        return messages
    prefix, anchor, body = messages[:fi], messages[fi], messages[fi + 1 :]
    snap = known_environment_message(known_files)
    mid_snap = [snap] if snap else []

    def pack_body(b: list[dict]) -> list[dict]:
        if len(b) <= tail_n:
            return [*mid_snap, *_dedupe_tool_results_keep_last(b)]
        middle, tail = b[:-tail_n], b[-tail_n:]
        return [*_dedupe_tool_results_keep_last(middle), *mid_snap, *tail]

    return [*prefix, anchor, *pack_body(body)]
