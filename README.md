# gigachat-openai-proxy

Тонкий прокси **OpenAI Chat Completions → GigaChat**: клиент шлёт привычный JSON, сервис ходит в GigaChat с OAuth и отдаёт ответ в формате, совместимом с OpenAI (удобно для Continue и других инструментов).

## Возможности

- Один синхронный маршрут: `POST /v1/chat/completions`
- Маппинг модели: вход `model: gigachat` (или любой другой) → upstream `GigaChat:latest` (настраивается)
- Кеш access token (~30 минут) с обновлением по `expires_at`
- Без streaming, tools, function calling и embeddings

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

Можно положить значения в файл `.env` в корне проекта (он в `.gitignore`).

Запуск сервера:

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
