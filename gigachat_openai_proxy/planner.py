import json

from gigachat_openai_proxy.mapping import upstream_body
from gigachat_openai_proxy.router import filter_gigachat_messages

# Текст не показывается пользователю напрямую — уходит в промпт основной модели при срыве парсера.
PLANNER_FALLBACK_ANSWER = (
    "Не удалось корректно распарсить план. Попробуй переформулировать запрос."
)

ACTION_PLANNER_FAILED = "planner_failed"

PLAN_DIRECT_TOOLS = frozenset({"read_file", "read_dir", "write_file", "patch"})

PLANNER_SYSTEM = (
    "Ты планировщик. Нужны read_file/read_dir/write_file/patch или достаточно final с текстом пользователю.\n"
    "В диалоге могут быть user-сообщения с префиксом [tool_result] — JSON с полями tool, args, result; "
    "учитывай их при следующем решении.\n"
    "Если в user-тексте уже есть блок ```имя_файла с новой строкой и далее содержимое того же файла, "
    "не выбирай read_file для этого пути — верни action=final и ответ по уже приведённому содержимому.\n"
    "Если read_file вернул not found, а раньше в user был такой фенс с тем же именем файла — снова final по фенсу, "
    "не повторяй read_file с тем же путём.\n"
    "write_file: args.path, args.content — полная перезапись UTF-8.\n"
    "patch: args.path, args.old_string (непустая), args.new_string; опционально args.replace_all (bool) — "
    "одна замена или все вхождения; при нескольких совпадениях без replace_all инструмент вернёт ошибку.\n"
    "Только action: final, tool, read_file, read_dir, write_file, patch (иные запрещены).\n"
    "Ты ОБЯЗАН отвечать только одним валидным JSON-объектом. Запрещено: любой текст вне JSON, комментарии, markdown. "
    "Если не уверен — всё равно верни JSON (например action=final с пояснением в answer). "
    "При нарушении формата запрос будет отклонён.\n"
    'Схема: {"action":"tool"|"final"|"read_file"|"read_dir"|"write_file"|"patch","tool":string|null,"args":object|null,"answer":string|null}\n'
    "Сокращение: action read_file/read_dir/write_file/patch с полем args (без tool).\n"
    "При action=final поле answer — готовый ответ пользователю (строка).\n"
    "При action=tool обязательны tool (read_file, read_dir, write_file или patch) и args (объект)."
)

_RETRY_USER = (
    "Ответ невалиден. Верни заново ТОЛЬКО один JSON без текста вокруг, по схеме: "
    '{"action":"tool"|"final"|"read_file"|"read_dir"|"write_file"|"patch",...}'
)


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
    if action == "final":
        ans = d.get("answer")
        if ans is None:
            raise ValueError("answer")
        return {"action": "final", "answer": str(ans), "tool": None, "args": {}}
    if action == "tool":
        tool, args = d.get("tool"), d.get("args")
        if not tool or not isinstance(tool, str) or tool not in PLAN_DIRECT_TOOLS:
            raise ValueError("tool")
        if not isinstance(args, dict):
            raise ValueError("args")
        return {"action": "tool", "tool": tool, "args": args, "answer": ""}
    if action in PLAN_DIRECT_TOOLS:
        args = d.get("args")
        if not isinstance(args, dict):
            raise ValueError("args")
        return {"action": action, "tool": None, "args": args, "answer": ""}
    raise ValueError("action")


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
