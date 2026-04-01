import asyncio
import json
import logging
import time
import uuid

import httpx

from gigachat_openai_proxy.mapping import auth_header
from gigachat_openai_proxy.settings import Settings, ssl_arg

_log = logging.getLogger(__name__)


def _json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class GigachatClient:
    def __init__(self, s: Settings, http: httpx.AsyncClient | None = None) -> None:
        self._s = s
        v = ssl_arg(s)
        self._own = http is None
        self._http = http or httpx.AsyncClient(timeout=s.timeout_sec, verify=v)
        self._lock = asyncio.Lock()
        self._token: str | None = None
        self._exp = 0.0

    async def aclose(self) -> None:
        if self._own:
            await self._http.aclose()

    async def _fetch_token(self) -> None:
        if self._s.gigachat_proxy_debug:
            _log.info(
                "gigachat oauth request POST %s scope=%s",
                self._s.gigachat_oauth_url,
                self._s.gigachat_scope,
            )
        h = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": auth_header(self._s.gigachat_authorization_key),
        }
        r = await self._http.post(
            self._s.gigachat_oauth_url, headers=h, data={"scope": self._s.gigachat_scope}
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        exp = data.get("expires_at")
        self._exp = float(exp) if exp else time.time() + 1800
        if self._s.gigachat_proxy_debug:
            _log.info("gigachat oauth ok expires_at=%s", self._exp)

    async def bearer(self) -> str:
        async with self._lock:
            if self._token and time.time() < self._exp - self._s.token_skew_sec:
                return self._token
            await self._fetch_token()
            assert self._token
            return self._token

    async def chat(self, body: dict) -> dict:
        tok = await self.bearer()
        url = f"{self._s.gigachat_api_base.rstrip('/')}/chat/completions"
        if self._s.gigachat_proxy_debug:
            _log.info("gigachat chat request POST %s %s", url, _json(body))
        r = await self._http.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"},
        )
        if self._s.gigachat_proxy_debug:
            _log.info("gigachat chat response status=%s %s", r.status_code, r.text)
        r.raise_for_status()
        return r.json()
