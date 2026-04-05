import json

from gigachat_openai_proxy.mapping import upstream_body
from gigachat_openai_proxy.router import filter_gigachat_messages
from gigachat_openai_proxy.tools import TOOL_REGISTRY

PLANNER_FALLBACK_ANSWER = (
    "Не удалось корректно распарсить план. Попробуй переформулировать запрос."
)

ACTION_PLANNER_FAILED = "planner_failed"

TOOL_RESULT_PREFIX = "[tool_result] "

PLANNER_PROMPT_TEMPLATE = (
    "Ты планировщик. Нужны инструменты ({available_actions}) или достаточно final с текстом пользователю.\n"
    "В диалоге могут быть user-сообщения с префиксом [tool_result] — JSON с полями tool, args, result; "
    "учитывай их при следующем решении.\n"
    "Если в user-тексте уже есть блок ```имя_файла с новой строки и далее содержимое того же файла, "
    "не выбирай read_file для этого пути — верни action=final и ответ по уже приведённому содержимому.\n"
    "Если read_file вернул not found, а раньше в user был такой фенс с тем же именем файла — снова final по фенсу, "
    "не повторяй read_file с тем же путём.\n"
    "write_file: args.path, args.content — полная перезапись UTF-8.\n"
    "patch: args.path, args.old_string (непустая), args.new_string; опционально args.replace_all (bool) — "
    "одна замена или все вхождения; при нескольких совпадениях без replace_all инструмент вернёт ошибку.\n"
    "action: final (ответ в answer), tool (поля tool+args) или короткая форма с action из списка инструментов. "
    "Неизвестное имя инструмента сервер отметит [tool_error] — перепланируй.\n"
    "Ты ОБЯЗАН отвечать только одним валидным JSON-объектом. Запрещено: любой текст вне JSON, комментарии, markdown. "
    "Если не уверен — всё равно верни JSON (например action=final с пояснением в answer). "
    "При нарушении формата запрос будет отклонён.\n"
    "Схема: объект с полями action (строка), при final — answer (строка); при tool — tool (строка) и args (объект); "
    "при короткой форме инструмента — action и args (объект). Имена инструментов: {available_actions}.\n"
    "При action=final поле answer — готовый ответ пользователю (строка).\n"
    "При action=tool обязательны tool (строка) и args (объект)."
)


def _available_actions() -> str:
    return "|".join(TOOL_REGISTRY.keys())


PLANNER_SYSTEM = PLANNER_PROMPT_TEMPLATE.format(available_actions=_available_actions())
_RETRY_USER = (
    "Ответ невалиден. Верни заново ТОЛЬКО один JSON без текста вокруг, "
    'по схеме: {"action":"read_dir","args":{"path":"..."}}'
)


def _tool_args_key(tool: str, args: dict) -> str:
    return json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False)


def state_has_tool_result(state: list[dict], tool: str, args: dict) -> bool:
    want = _tool_args_key(tool, args)
    for m in state:
        if m.get("role") != "user":
            continue
        c = str(m.get("content") or "")
        if not c.startswith(TOOL_RESULT_PREFIX):
            continue
        try:
            body = json.loads(c[len(TOOL_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if not isinstance(body, dict):
            continue
        t, a = body.get("tool"), body.get("args")
        if not isinstance(t, str) or not isinstance(a, dict):
            continue
        if _tool_args_key(t, a) == want:
            return True
    return False


def tool_result_identity_key(message: dict) -> str | None:
    if message.get("role") != "user":
        return None
    c = str(message.get("content") or "")
    if not c.startswith(TOOL_RESULT_PREFIX):
        return None
    try:
        body = json.loads(c[len(TOOL_RESULT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    t, a = body.get("tool"), body.get("args")
    if not isinstance(t, str) or not isinstance(a, dict):
        return None
    return _tool_args_key(t, a)


def tool_result_message(tool: str, args: dict, result: str) -> dict:
    payload = json.dumps({"tool": tool, "args": args, "result": result}, ensure_ascii=False)
    return {"role": "user", "content": f"{TOOL_RESULT_PREFIX}{payload}"}


def effective_tool_name(plan: dict) -> str:
    return str(plan.get("tool") or "") if plan.get("action") == "tool" else str(plan.get("action") or "")


def _coerce_args(v) -> dict:
    return v if isinstance(v, dict) else {}


def _strip_markdown_fence(t: str) -> str:
    if "```" not in t:
        return t
    i = t.find("```")
    j = t.find("\n", i)
    t = t[j + 1 :] if j != -1 else t[i + 3 :]
    k = t.find("```")
    if k != -1:
        t = t[:k]
    return t.strip()


def _extract_json_object(text: str) -> dict:
    t = _strip_markdown_fence((text or "").strip())
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(t, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no json object")


def parse_plan(text: str) -> dict:
    d = _extract_json_object(text)
    action = d.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action")
    action = action.strip()
    if action == "final":
        ans = d.get("answer")
        if ans is None:
            raise ValueError("answer")
        return {"action": "final", "answer": str(ans), "tool": None, "args": {}}
    if action == "tool":
        tool = d.get("tool")
        if not tool or not isinstance(tool, str) or not tool.strip():
            raise ValueError("tool")
        args = d.get("args")
        if not isinstance(args, dict):
            raise ValueError("args")
        return {"action": "tool", "tool": tool.strip(), "args": args, "answer": ""}
    return {"action": action, "tool": None, "args": _coerce_args(d.get("args")), "answer": ""}


def planner_messages(dialog_messages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": PLANNER_SYSTEM}, *filter_gigachat_messages(dialog_messages)]


async def run_planner(
    gc, req: dict, gigachat_model: str, *, dialog: list[dict], max_attempts: int = 3
) -> dict:
    msgs = [{"role": "system", "content": PLANNER_SYSTEM}, *dialog]
    for _ in range(max_attempts):
        raw = await gc.chat(upstream_body({**req, "messages": msgs}, gigachat_model, planner=True))
        text = (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        try:
            return parse_plan(str(text))
        except Exception:
            msgs.append({"role": "assistant", "content": str(text)})
            msgs.append({"role": "user", "content": _RETRY_USER})
    return {
        "action": ACTION_PLANNER_FAILED,
        "answer": "",
        "tool": None,
        "args": {},
    }
