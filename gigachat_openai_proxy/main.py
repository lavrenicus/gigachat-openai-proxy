import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gigachat_openai_proxy.client import GigachatClient
from gigachat_openai_proxy.mapping import openai_from_ollama, openai_response, upstream_body
from gigachat_openai_proxy.ollama_client import ollama_chat
from gigachat_openai_proxy.router import filter_gigachat_messages, use_ollama
from gigachat_openai_proxy.settings import Settings


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
) -> JSONResponse:
    try:
        dump = req.model_dump()
        if use_ollama(req.messages):
            raw_o = await ollama_chat(ohttp, s, req.messages, dump)
            return JSONResponse(openai_from_ollama(raw_o, req.model))
        dump["messages"] = filter_gigachat_messages(dump["messages"])
        raw = await gc.chat(upstream_body(dump, s.gigachat_model))
        return JSONResponse(openai_response(raw, req.model))
    except httpx.HTTPStatusError as e:
        detail = e.response.text
        try:
            detail = e.response.json()
        except Exception:
            pass
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e))
