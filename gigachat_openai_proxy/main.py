import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from gigachat_openai_proxy.agent_context import (
    SOFT_STEP_LIMIT_BEFORE_NUDGE,
    compress_planner_state,
    count_recent_failed_patch_results,
    first_user_task_text,
    internal_critic_final_without_mutation,
    record_tool_for_known_files,
    soft_resource_nudge_message,
    verify_final_against_edit_intent,
)
from gigachat_openai_proxy.app_config import AppConfig
from gigachat_openai_proxy.client import GigachatClient
from gigachat_openai_proxy.inline_file import inline_file_body_for_path
from gigachat_openai_proxy.mapping import openai_from_text, openai_response, upstream_body
from gigachat_openai_proxy.planner import (
    ACTION_PLANNER_FAILED,
    PLANNER_FALLBACK_ANSWER,
    effective_tool_name,
    run_planner,
    state_has_tool_result,
    tool_result_message,
)
from gigachat_openai_proxy.tools import TOOL_REGISTRY
from gigachat_openai_proxy.pipeline import Envelope, Pipeline
from gigachat_openai_proxy.router import filter_gigachat_messages
from gigachat_openai_proxy.settings import Settings
from gigachat_openai_proxy.tls import verify_arg
from gigachat_openai_proxy.debug import pretty_openai_chat_response

_MAX_AGENT_STEPS = 72
_MAX_CONSECUTIVE_DUPLICATE_SKIPS = 5
_MAX_INTERNAL_CRITIC_ON_FINAL = 3
_LOG_RESULT_LEN = 4000


def _stuck_duplicate_loop_user_message() -> dict:
    return {
        "role": "user",
        "content": (
            "[tool_error] stuck in loop: same tool+args repeated without progress; "
            "use a different tool/path or action=final with an answer."
        ),
    }


class ChatReq(BaseModel):
    model: str = "gigachat"
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    tools: list[dict] | None = None


_TITLE_HINTS = (
    "title for the chat",
    "title for a conversation",
    "descriptive title",
    "reply with a title",
    "generate a short",
    "3-4 words",
    "3-7 words",
    "chat title",
    "words in length",
    "given the following",
    "название чата",
    "заголовок чата",
)


def _is_title_generation_request(req: ChatReq) -> bool:
    mt = req.max_tokens
    if mt is None or mt > 128:
        return False
    blob = " ".join(
        str(m.get("content") or "").lower()
        for m in req.messages
        if m.get("role") in ("user", "system")
    )
    return any(h in blob for h in _TITLE_HINTS)


def _trunc_log(s: str, limit: int = _LOG_RESULT_LEN) -> str:
    return s if len(s) <= limit else f"{s[:limit]}\n...[truncated]"


async def _gigachat_delegate_planner_fail(
    gc: GigachatClient, req: ChatReq, dump: dict, gigachat_model: str
) -> dict:
    sys = (
        "Служебно: внутренний JSON-планировщик не вернул валидный ответ после нескольких попыток.\n"
        f"Ориентир по смыслу ответа пользователю (не цитируй дословно): {PLANNER_FALLBACK_ANSWER}\n"
        "Сформулируй один ответ по переписке ниже своими словами, без JSON, без слов «планировщик» и без технических деталей."
    )
    msgs = [{"role": "system", "content": sys}, *filter_gigachat_messages(req.messages)]
    raw = await gc.chat(upstream_body({**dump, "messages": msgs}, gigachat_model))
    return openai_response(raw, req.model)


async def _planner_agent(
    gc: GigachatClient, req: ChatReq, dump: dict, gigachat_model: str, *, debug: bool = False
) -> dict:
    _lg = logging.getLogger("gigachat_openai_proxy")
    state = filter_gigachat_messages(req.messages)
    task_text = first_user_task_text(state)
    consecutive_dup_skips = 0
    known_files: dict[str, str] = {}
    soft_nudge_sent = False
    final_critic_rounds = 0
    for step in range(_MAX_AGENT_STEPS):
        if step == SOFT_STEP_LIMIT_BEFORE_NUDGE and not soft_nudge_sent:
            state = [*state, soft_resource_nudge_message()]
            soft_nudge_sent = True
        planning_dialog = compress_planner_state(state, known_files=known_files)
        plan = await run_planner(gc, dump, gigachat_model, dialog=planning_dialog)
        if debug:
            _lg.info("agent step=%s plan=%s", step, _trunc_log(json.dumps(plan, ensure_ascii=False)))
        if plan["action"] == ACTION_PLANNER_FAILED:
            if debug:
                _lg.info("agent step=%s planner parse failed; delegating to gigachat", step)
            return await _gigachat_delegate_planner_fail(gc, req, dump, gigachat_model)
        if plan["action"] == "final":
            ans = str(plan.get("answer") or "")
            if debug:
                _lg.info("agent step=%s final_answer=%s", step, _trunc_log(ans))
            if verify_final_against_edit_intent(plan, task_text, known_files):
                return openai_from_text(ans, req.model)
            if final_critic_rounds >= _MAX_INTERNAL_CRITIC_ON_FINAL:
                if debug:
                    _lg.info("agent step=%s final forced after internal critic budget", step)
                return openai_from_text(ans, req.model)
            final_critic_rounds += 1
            state = [
                *state,
                internal_critic_final_without_mutation(
                    draft=ans,
                    known_files=known_files,
                    failed_patches=count_recent_failed_patch_results(state),
                    step=step,
                    max_steps=_MAX_AGENT_STEPS,
                ),
            ]
            if debug:
                _lg.info("agent step=%s internal_critic injected (edit intent without write/patch)", step)
            continue
        name = effective_tool_name(plan)
        args = plan["args"] if isinstance(plan.get("args"), dict) else {}
        if name in TOOL_REGISTRY:
            if state_has_tool_result(state, name, args):
                consecutive_dup_skips += 1
                if debug:
                    _lg.info(
                        "agent step=%s skip duplicate tool_result tool=%s consecutive=%s",
                        step,
                        name,
                        consecutive_dup_skips,
                    )
                if consecutive_dup_skips >= _MAX_CONSECUTIVE_DUPLICATE_SKIPS:
                    state = [*state, _stuck_duplicate_loop_user_message()]
                    consecutive_dup_skips = 0
                    if debug:
                        _lg.info("agent step=%s injected stuck_duplicate_loop hint", step)
                continue
            consecutive_dup_skips = 0
            _lg.info("Mapped action '%s' → tool='%s'", name, name)
            if name == "read_file":
                pth = str(args.get("path") or "")
                inl = inline_file_body_for_path(state, pth)
                res = inl if inl is not None else TOOL_REGISTRY[name](args)
            else:
                res = TOOL_REGISTRY[name](args)
            msg = tool_result_message(name, args, res)
            record_tool_for_known_files(name, args, res, known_files)
            if debug:
                _lg.info("agent step=%s tool_result=%s", step, _trunc_log(msg["content"]))
            state = [*state, msg]
        else:
            err = (
                f'[tool_error] unknown action "{name}", '
                f"available: {list(TOOL_REGISTRY.keys())}"
            )
            if debug:
                _lg.info("agent step=%s %s", step, _trunc_log(err))
            consecutive_dup_skips = 0
            state = [*state, {"role": "user", "content": err}]
    if debug:
        _lg.info("agent stopped: max_steps=%s", _MAX_AGENT_STEPS)
    return openai_from_text("Лимит шагов планировщика исчерпан.", req.model)


async def _direct_title_chat(
    gc: GigachatClient, req: ChatReq, dump: dict, gigachat_model: str
) -> dict:
    raw = await gc.chat(upstream_body({**dump, "messages": list(req.messages)}, gigachat_model))
    return openai_response(raw, req.model)


def _setup_proxy_debug_logging() -> None:
    lg = logging.getLogger("gigachat_openai_proxy")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        lg.addHandler(h)
    lg.propagate = False


def gc_client(request: Request) -> GigachatClient:
    return request.app.state.gc


def settings(request: Request) -> Settings:
    return request.app.state.settings


def config(request: Request) -> AppConfig:
    return request.app.state.config


def ollama_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.ollama_http


async def pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def _verify_arg(s: Settings, c: AppConfig):
    if c.use_mincifry_ca:
        from gigachat_openai_proxy.mincifry_ca import ensure_ca_bundle

        ensure_ca_bundle(c.mincifry_ca_path, url=c.mincifry_ca_url)
    return verify_arg(c.verify_ssl, c.ca_bundle)


def create_app(s: Settings | None = None, c: AppConfig | None = None) -> FastAPI:
    cfg = c or AppConfig()
    st = s or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.debug:
            _setup_proxy_debug_logging()
        app.state.settings = st
        app.state.config = cfg
        app.state.gc = GigachatClient(st, verify=_verify_arg(st, cfg), debug=cfg.debug)
        app.state.ollama_http = httpx.AsyncClient(timeout=st.ollama_timeout_sec)
        app.state.pipeline = Pipeline(queue_size=cfg.pipeline_queue_size, workers=cfg.pipeline_workers)
        await app.state.pipeline.start()
        yield
        await app.state.pipeline.aclose()
        await app.state.ollama_http.aclose()
        await app.state.gc.aclose()

    app = FastAPI(title="GigaChat OpenAI proxy", lifespan=lifespan)
    _mount_routes(app)
    return app


def _mount_routes(app: FastAPI) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatReq,
        gc: GigachatClient = Depends(gc_client),
        s: Settings = Depends(settings),
        cfg: AppConfig = Depends(config),
    ) -> Any:
        try:
            dump = req.model_dump()
            if cfg.debug:
                logging.getLogger("gigachat_openai_proxy").info(
                    "proxy incoming /v1/chat/completions %s", pretty_openai_chat_response(dump)
                )
            wants_stream = bool(req.stream)
            out = (
                await _direct_title_chat(gc, req, dump, s.gigachat_model)
                if _is_title_generation_request(req)
                else await _planner_agent(gc, req, dump, s.gigachat_model, debug=cfg.debug)
            )
            if cfg.debug:
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

    @app.post("/v1/pipeline/ingest")
    async def pipeline_ingest(
        env: Envelope,
        request: Request,
        cfg: AppConfig = Depends(config),
        pl: Pipeline = Depends(pipeline),
    ) -> dict:
        _check_generator_token(request, cfg)
        return {"id": pl.submit(env)}

    @app.get("/v1/pipeline/result/{id}")
    async def pipeline_result(id: str, pl: Pipeline = Depends(pipeline)) -> dict:
        r = pl.get(id)
        if not r:
            raise HTTPException(status_code=404, detail="not found")
        return r.model_dump()

    @app.post("/v1/pipeline/process")
    async def pipeline_process(env: Envelope, pl: Pipeline = Depends(pipeline), timeout_sec: float = 30.0) -> dict:
        pid = pl.submit(env)
        r = await pl.wait(pid, timeout_sec=timeout_sec)
        assert r
        return r.model_dump()


app = create_app()


def _check_generator_token(req: Request, cfg: AppConfig) -> None:
    tok = cfg.pipeline_generator_token
    if not tok:
        return
    got = req.headers.get("x-generator-token")
    if got != tok:
        raise HTTPException(status_code=401, detail="bad generator token")


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
