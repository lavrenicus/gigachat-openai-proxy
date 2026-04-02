import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)


class Envelope(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class PipelineResult(BaseModel):
    id: str
    status: str  # pending|ok|error
    output: Envelope | None = None
    error: str | None = None


@dataclass(frozen=True)
class _Item:
    id: str
    env: Envelope


def _now() -> int:
    return int(time.time())


def _clean_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def process_env(env: Envelope) -> Envelope:
    md = dict(env.metadata or {})
    if md.get("cause_error"):
        raise ValueError("processor_error")
    content = _clean_text(env.content)
    md.setdefault("type", "text")
    md["processed_at"] = _now()
    md["content_len"] = len(content)
    return Envelope(content=content, metadata=md, timestamp=env.timestamp)


class Pipeline:
    def __init__(self, queue_size: int = 0, workers: int = 1) -> None:
        self._q: asyncio.Queue[_Item] = asyncio.Queue(maxsize=max(0, int(queue_size)))
        self._ev: dict[str, asyncio.Event] = {}
        self._res: dict[str, PipelineResult] = {}
        self._tasks: list[asyncio.Task] = []
        self._workers = max(1, int(workers))
        self._closed = False

    async def start(self) -> None:
        if self._tasks:
            return
        self._closed = False
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self._workers)]

    async def aclose(self) -> None:
        self._closed = True
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    def submit(self, env: Envelope, id: str | None = None) -> str:
        pid = id or uuid.uuid4().hex
        self._res[pid] = PipelineResult(id=pid, status="pending")
        self._ev[pid] = asyncio.Event()
        self._q.put_nowait(_Item(id=pid, env=env))
        return pid

    def get(self, id: str) -> PipelineResult | None:
        return self._res.get(id)

    async def wait(self, id: str, timeout_sec: float) -> PipelineResult | None:
        ev = self._ev.get(id)
        if not ev:
            return None
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            return self._res.get(id)
        return self._res.get(id)

    async def _worker(self, idx: int) -> None:
        while not self._closed:
            it = await self._q.get()
            try:
                out = process_env(it.env)
                self._res[it.id] = PipelineResult(id=it.id, status="ok", output=out)
            except Exception as e:
                _log.exception("pipeline worker=%s id=%s error", idx, it.id)
                self._res[it.id] = PipelineResult(id=it.id, status="error", error=str(e))
            finally:
                ev = self._ev.get(it.id)
                if ev:
                    ev.set()
                self._q.task_done()
