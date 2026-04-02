import json
import logging

import httpx

from gigachat_openai_proxy.settings import Settings

_log = logging.getLogger(__name__)


def _json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


async def ollama_chat(
    http: httpx.AsyncClient, s: Settings, messages: list[dict], req: dict
) -> dict:
    url = f"{s.ollama_base.rstrip('/')}/api/chat"
    payload: dict = {"model": s.ollama_model, "messages": messages, "stream": False}
    t, mt = req.get("temperature"), req.get("max_tokens")
    opt: dict = {}
    if t is not None:
        opt["temperature"] = t
    if mt is not None:
        opt["num_predict"] = mt
    if opt:
        payload["options"] = opt
    if s.gigachat_proxy_debug:
        _log.info("ollama request POST %s %s", url, _json(payload))
    r = await http.post(url, json=payload)
    if s.gigachat_proxy_debug:
        _log.info("ollama response status=%s %s", r.status_code, r.text)
    r.raise_for_status()
    return r.json()
