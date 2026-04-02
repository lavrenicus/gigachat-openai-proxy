# gigachat-openai-proxy

Тонкий прокси **OpenAI Chat Completions**: обычный диалог идёт в **GigaChat**, запросы с признаками tool/агента — в **Ollama** (локально). Ответ всегда в формате OpenAI, Continue видит одну модель.

## Возможности

- Один синхронный маршрут: `POST /v1/chat/completions`
- Роутинг: эвристика по тексту сообщений (маркеры вроде `read_file`, `прочитай файл`, …) или `system` с подстрокой `tool` → **только Ollama**; иначе → **только GigaChat** (без смешивания в одном запросе)
- Перед GigaChat из списка сообщений убираются роли `system`, чтобы не провоцировать лишние tool-ответы
- GigaChat: `GIGACHAT_MODEL` (по умолчанию `GigaChat:latest`), OAuth и кеш токена
- Ollama: `POST {OLLAMA_BASE}/api/chat`, модель `OLLAMA_MODEL` (по умолчанию `qwen2.5-coder:7b`)
- Без streaming и без отдельного OpenAI tool-calling протокола на стороне прокси

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
| `VERIFY_SSL` | `true` / `false` — при проблемах с корпоративным CA |
| `CA_BUNDLE` | Путь к файлу CA вместо системного хранилища |
| `TIMEOUT_SEC` | Таймаут HTTP к GigaChat и OAuth (по умолчанию `120`) |
| `GIGACHAT_PROXY_DEBUG` | `true` / `1` — в лог (INFO) тела запросов/ответов к GigaChat и Ollama; для OAuth — только URL и `scope` |
| `OLLAMA_BASE` | База Ollama (по умолчанию `http://localhost:11434`) |
| `OLLAMA_MODEL` | Модель для ветки tools (по умолчанию `qwen2.5-coder:7b`) |
| `OLLAMA_TIMEOUT_SEC` | Таймаут HTTP к Ollama (по умолчанию `120`) |

Можно положить значения в файл `.env` в корне проекта (он в `.gitignore`).

Запуск сервера:

```bash
poetry run serve
```

Отладочные логи upstream без правки `.env`:

```bash
poetry run serve --debug
```

(перед стартом выставляется `GIGACHAT_PROXY_DEBUG=true` в окружении процесса.)

Флаг ищется как отдельный аргумент `--debug` в `sys.argv`, поэтому он срабатывает даже при предупреждении Poetry про «script is not installed». При включённом debug для логгера `gigachat_openai_proxy` добавляется вывод в stderr, чтобы строки upstream не терялись рядом с логами uvicorn.

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

## Замечания

- **TLS**: при ошибках сертификата используйте корневой CA (`CA_BUNDLE`) или отключите проверку только осознанно (`VERIFY_SSL=false`).
- **Доступ из‑за границы**: API GigaChat ориентирован на РФ; при необходимости поднимайте прокси на VPS в подходящей зоне.
- **Контекст**: прокси stateless — весь диалог должен приходить в `messages`, как у OpenAI.

## Лицензия

Укажите лицензию при необходимости.
