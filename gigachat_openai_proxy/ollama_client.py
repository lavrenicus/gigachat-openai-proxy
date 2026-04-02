import json
import logging

import httpx

from gigachat_openai_proxy.settings import Settings
from gigachat_openai_proxy.debug import pretty, pretty_text

_log = logging.getLogger(__name__)


def _json(obj: object) -> str:
    return pretty(obj)


async def ollama_chat(
    http: httpx.AsyncClient, s: Settings, messages: list[dict], req: dict, debug: bool = False
) -> dict:
    url = f"{s.ollama_base.rstrip('/')}/api/chat"
    payload: dict = {"model": s.ollama_model, "messages": messages, "stream": False}
    tools = req.get("tools")
    if isinstance(tools, list) and tools:
        payload["tools"] = tools
    t, mt = req.get("temperature"), req.get("max_tokens")
    opt: dict = {}
    if t is not None:
        opt["temperature"] = t
    if mt is not None:
        opt["num_predict"] = mt
    if opt:
        payload["options"] = opt
    if debug:
        _log.info("ollama request POST %s\n%s", url, _json(payload))
    try:
        r = await http.post(url, json=payload)
    except httpx.HTTPError as e:
        if debug:
            _log.error("ollama request failed: %s", e)
        raise
    if debug:
        _log.info(
            "ollama response status=%s\n%s",
            r.status_code,
            pretty_text(r.text),
        )
    r.raise_for_status()
    return r.json()
