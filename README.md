# gigachat-openai-proxy

Тонкий прокси **OpenAI Chat Completions**: внешний цикл **GigaChat (Planner)** → при вызове инструмента результат попадает в контекст как **user** с префиксом **`[tool_result]`** и JSON → снова Planner, пока не будет **`final`**. Исполняемые инструменты задаются реестром **`TOOL_REGISTRY`** в `tools.py` (по умолчанию **`read_file`**, **`read_dir`**, **`write_file`**, **`patch`**); неизвестный `action` даёт **`[tool_error]`** и перепланирование. Для вызова GigaChat диалог **сжимается** (якорь задачи, дедуп `tool_result` в «середине», хвост последних сообщений, блок **`[Known Environment]`** по успешным чтениям/записям). Перед отдачей **`final`** при явном намерении менять файлы проверяется наличие успешного **write/patch** в снимке; иначе в контекст добавляется **`[internal_critic]`** (до лимита попыток). Есть защита от петель: пропуск дубликатов `tool+args`, «застревание» на дубликатах, мягкий nudge к `final` на высоком номере шага. Инструменты выполняются детерминированно на стороне прокси (без Ollama в оркестрации). Ответ в формате OpenAI; Continue видит одну модель.

## Возможности

- Один синхронный маршрут: `POST /v1/chat/completions`
- Каждый запрос: **Planner-loop** (GigaChat, JSON-план: `final`, при `action=tool` поля `tool` и `args`, либо короткая форма `action` из реестра); шаг инструмента → **executor** → **`user`: `[tool_result] {...}`** (не второй `system`: у GigaChat **одно** `system` и **только первое** в массиве) → следующий вызов Planner. В GigaChat уходит **сжатый** вариант истории (полный `state` хранится для инструментов и дедупа). Вызовы планировщика с **собственным** `max_tokens`: не ниже **512**, по умолчанию **2048** (клиентский низкий лимит, например `30` для заголовка чата, на планировщик не распространяется)
- Запросы **генерации заголовка чата** (эвристика: небольшой `max_tokens` ≤128 + типичные фразы в `user`/`system`, как у Continue) обходят планировщик и идут **одним** прямым вызовом GigaChat с исходным лимитом клиента
- Поле `tools` в запросе **не обязательно** для работы инструментов (схемы клиента можно игнорировать; исполняются только `read_file`, `read_dir`, `write_file` и `patch` на стороне прокси — не терминал, не браузер и т.д.)
- Перед первым шагом из пользовательских сообщений убираются клиентские `system` (как раньше); служебные `[tool_result]` идут ролью `user` и не ломают порядок для API
- GigaChat: `GIGACHAT_MODEL` (по умолчанию `GigaChat:latest`), OAuth и кеш токена
- Ollama в HTTP-цикле чата **не используется** (клиент в приложении всё ещё создаётся для совместимости; см. `ollama_tools.ollama_tool_loop` при необходимости отдельно)
- **stream: true**: минимальная отдача одним чанком в SSE (удобно для Continue); отдельного полноценного OpenAI tool-calling на стороне прокси нет
- **`write_file`** и **`patch`** изменяют файлы на машине, где запущен прокси, по путям от модели — используйте только в доверенной среде

## Требования

- Python 3.10+
- [Poetry](https://python-poetry.org/)

## Установка и запуск

```bash
poetry install
```

Переменные окружения (обязательно задать ключ из [кабинета GigaChat](https://developers.sber.ru/)):

| Переменная | Описание |
|------------|----------|
| `GIGACHAT_AUTHORIZATION_KEY` | Ключ авторизации (Base64; в заголовок подставляется как `Basic …`, если префикса нет) |
| `GIGACHAT_SCOPE` | По умолчанию `GIGACHAT_API_PERS` |
| `TIMEOUT_SEC` | Таймаут HTTP к GigaChat и OAuth (по умолчанию `120`) |
| `OLLAMA_BASE` | База Ollama (по умолчанию `http://localhost:11434`) |
| `OLLAMA_MODEL` | Модель для ветки tools (по умолчанию `qwen2.5-coder:7b`) |
| `OLLAMA_TIMEOUT_SEC` | Таймаут HTTP к Ollama (по умолчанию `120`) |

Можно положить значения в файл `.env` в корне проекта (он в `.gitignore`).

Запуск сервера:

```bash
poetry run serve
```

Отладочные логи upstream (тела запросов/ответов к GigaChat и Ollama):

```bash
poetry run serve --debug
```

Флаги TLS для `serve` (не через `.env`): `--mincifry-ca` (докачать PEM при старте), `--no-verify-ssl` (только если осознанно нужно обойти проверку).

Флаг ищется как отдельный аргумент `--debug` в `sys.argv`, поэтому он срабатывает даже при предупреждении Poetry про «script is not installed». При включённом debug для логгера `gigachat_openai_proxy` добавляется вывод в stderr, чтобы строки upstream не терялись рядом с логами uvicorn. Дополнительно пишутся шаги агента: `plan=...`, `tool_result=...`, пропуск дубликата, `internal_critic`, `final_answer=...` (длинные строки обрезаются).

Команда `serve` — это entry point из `pyproject.toml`; он появляется в venv только после установки **самого проекта**. Сделайте из корня репозитория:

```bash
poetry install
```

Без флага `--no-root` (по умолчанию корневой пакет ставится). Если в prompt активирован другой интерпретатор (например Conda `(proxy_env)`), Poetry может путаться с окружением: в репозитории включён `poetry.toml` с `prefer-active-existing = false`, чтобы `poetry run` использовал venv Poetry. После смены настроек снова выполните `poetry install`.

Если предупреждение про «script is not installed» остаётся или ставить проект не нужно, используйте запуск без консольного скрипта (зависимости `uvicorn` всё равно подтянуты):

```bash
poetry run uvicorn gigachat_openai_proxy.main:app --host 0.0.0.0 --port 8000
```

или:

```bash
poetry run python -m gigachat_openai_proxy
```

Пример запроса:

```json
{
  "model": "gigachat",
  "messages": [
    {"role": "system", "content": "Ты ассистент."},
    {"role": "user", "content": "Привет"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

## Continue

Фрагмент конфигурации:

```yaml
- name: GigaChat
  provider: openai
  apiBase: http://localhost:8000/v1
  model: gigachat
  roles:
    - chat
    - edit
```

## Тесты

```bash
poetry run pytest tests -q
```

## Сертификаты для GigaChat (официально, Сбер)

По документации: [Использование сертификатов Минцифры в GigaChat](https://developers.sber.ru/docs/ru/gigachat/certificates) — нужны корневой и выпускающий из `gu-st.ru`.

Скачать в проект (в `certs/`, плюс общий `certs/ca.pem` для прокси):

```bash
poetry run fetch-gigachat-ca
```

URL в коде те же, что в доке:  
`https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt`,  
`https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt`.

Прокси **автоматически** подхватывает `certs/ca.pem`, иначе — `certs/mincifry.pem` (см. ниже).

## Альтернатива: корень с репозитория УЦ ГИС

```bash
poetry run fetch-mincifry-ca
```

Кладёт `certs/mincifry.pem`; для GigaChat предпочтительнее **`fetch-gigachat-ca`**.

## Замечания

- **TLS / GigaChat**: при `CERTIFICATE_VERIFY_FAILED` сначала выполните `poetry run fetch-gigachat-ca` и перезапустите прокси.
- **Доступ из‑за границы**: API GigaChat ориентирован на РФ; при необходимости поднимайте прокси на VPS в подходящей зоне.
- **Контекст**: прокси stateless — весь диалог должен приходить в `messages`, как у OpenAI.

## Лицензия

Укажите лицензию при необходимости.
