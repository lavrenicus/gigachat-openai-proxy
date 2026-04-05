import json
import re

from gigachat_openai_proxy.mapping import upstream_body
from gigachat_openai_proxy.router import filter_gigachat_messages

# Текст не показывается пользователю напрямую — уходит в промпт основной модели при срыве парсера.
PLANNER_FALLBACK_ANSWER = (
    "Не удалось корректно распарсить план. Попробуй переформулировать запрос."
)

ACTION_PLANNER_FAILED = "planner_failed"

PLAN_DIRECT_TOOLS = frozenset({"read_file", "read_dir"})

PLANNER_SYSTEM = (
    "Ты планировщик. Разбери задачу: нужны ли read_file/read_dir или достаточно ответа текстом.\n"
    "В диалоге могут быть user-сообщения с префиксом [tool_result] — JSON с полями tool, args, result; "
    "учитывай их при следующем решении.\n"
    "Ты ОБЯЗАН отвечать только одним валидным JSON-объектом. Запрещено: любой текст вне JSON, комментарии, markdown. "
    "Если не уверен — всё равно верни JSON (например action=final с пояснением в answer). "
    "При нарушении формата запрос будет отклонён.\n"
    'Схема: {"action":"tool"|"final"|"read_file"|"read_dir","tool":string|null,"args":object|null,"answer":string|null}\n'
    "Допустимо сокращение: action может быть read_file или read_dir с полем args (без поля tool).\n"
    "При action=final поле answer — готовый ответ пользователю (строка).\n"
    "При action=tool обязательны tool (read_file или read_dir) и args (объект)."
)

_RETRY_USER = (
    "Ответ невалиден. Верни заново ТОЛЬКО один JSON без текста вокруг, по схеме: "
    '{"action":"tool"|"final"|"read_file"|"read_dir",...}'
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
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(t[i : j + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("no json object")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("not object")
    return obj


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
        if not tool or not isinstance(tool, str):
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
        raw = await gc.chat(upstream_body({**req, "messages": msgs}, gigachat_model))
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
