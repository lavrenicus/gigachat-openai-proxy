from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gigachat_openai_proxy.client import GigachatClient
from gigachat_openai_proxy.mapping import openai_response, upstream_body
from gigachat_openai_proxy.settings import Settings


class ChatReq(BaseModel):
    model: str = "gigachat"
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    app.state.settings = s
    gc = GigachatClient(s)
    app.state.gc = gc
    yield
    await gc.aclose()


app = FastAPI(title="GigaChat OpenAI proxy", lifespan=lifespan)


def gc_client(request: Request) -> GigachatClient:
    return request.app.state.gc


def settings(request: Request) -> Settings:
    return request.app.state.settings


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatReq,
    gc: GigachatClient = Depends(gc_client),
    s: Settings = Depends(settings),
) -> JSONResponse:
    try:
        raw = await gc.chat(upstream_body(req.model_dump(), s.gigachat_model))
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
