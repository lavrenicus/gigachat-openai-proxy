import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from gigachat_openai_proxy.client import GigachatClient
from gigachat_openai_proxy.mapping import openai_from_ollama, openai_response, upstream_body
from gigachat_openai_proxy.ollama_client import ollama_chat
from gigachat_openai_proxy.router import filter_gigachat_messages, use_ollama
from gigachat_openai_proxy.settings import Settings
from gigachat_openai_proxy.debug import pretty_openai_chat_response


def _setup_proxy_debug_logging() -> None:
    lg = logging.getLogger("gigachat_openai_proxy")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        lg.addHandler(h)
    lg.propagate = False


class ChatReq(BaseModel):
    model: str = "gigachat"
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    tools: list[dict] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    if s.gigachat_proxy_debug:
        _setup_proxy_debug_logging()
    app.state.settings = s
    gc = GigachatClient(s)
    app.state.gc = gc
    app.state.ollama_http = httpx.AsyncClient(timeout=s.ollama_timeout_sec)
    yield
    await app.state.ollama_http.aclose()
    await gc.aclose()


app = FastAPI(title="GigaChat OpenAI proxy", lifespan=lifespan)


def gc_client(request: Request) -> GigachatClient:
    return request.app.state.gc


def settings(request: Request) -> Settings:
    return request.app.state.settings


def ollama_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.ollama_http


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatReq,
    gc: GigachatClient = Depends(gc_client),
    s: Settings = Depends(settings),
    ohttp: httpx.AsyncClient = Depends(ollama_http),
) -> Any:
    try:
        dump = req.model_dump()
        if s.gigachat_proxy_debug:
            logging.getLogger("gigachat_openai_proxy").info("proxy incoming /v1/chat/completions %s", pretty_openai_chat_response(dump))
        wants_stream = bool(req.stream)
        if use_ollama(req.messages):
            raw_o = await ollama_chat(ohttp, s, req.messages, dump)
            out = openai_from_ollama(raw_o, req.model)
            return (
                JSONResponse(out)
                if (not wants_stream or out["choices"][0]["message"].get("tool_calls"))
                else StreamingResponse(
                    _openai_chat_to_sse(out), media_type="text/event-stream"
                )
            )
        dump["messages"] = filter_gigachat_messages(dump["messages"])
        raw = await gc.chat(upstream_body(dump, s.gigachat_model))
        out = openai_response(raw, req.model)
        if s.gigachat_proxy_debug:
            logging.getLogger("gigachat_openai_proxy").info(
                "proxy returning OpenAI JSON\n%s", pretty_openai_chat_response(out)
            )
        return (
            JSONResponse(out)
            if not wants_stream
            else StreamingResponse(_openai_chat_to_sse(out), media_type="text/event-stream")
        )
    except httpx.HTTPStatusError as e:
        detail = e.response.text
        try:
            detail = e.response.json()
        except Exception:
            pass
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e))


def _openai_chat_to_sse(out: dict):
    # Минимальная поддержка stream:true для Continue:
    # отдаём весь content одним чанком, затем finish_reason и [DONE].
    c0 = out["choices"][0]
    msg = c0["message"]
    content = str(msg.get("content") or "")
    created = out["created"]
    base = {
        "id": out["id"],
        "object": "chat.completion.chunk",
        "created": created,
        "model": out["model"],
    }
    chunk1 = dict(base)
    chunk1["choices"] = [
        {
            "index": 0,
            # Continue (и OpenAI-compatible parsers) обычно смотрят только на delta.content
            "delta": {"content": content},
        }
    ]
    chunk2 = dict(base)
    chunk2["choices"] = [
        {"index": 0, "delta": {}, "finish_reason": c0.get("finish_reason") or "stop"},
    ]

    def gen():
        # SSE обычно ожидает CRLF-разделители
        yield f"data: {json.dumps(chunk1, ensure_ascii=False)}\r\n\r\n"
        yield f"data: {json.dumps(chunk2, ensure_ascii=False)}\r\n\r\n"
        yield "data: [DONE]\r\n\r\n"

    return gen()
