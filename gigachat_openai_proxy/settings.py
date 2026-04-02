from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gigachat_oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_api_base: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_model: str = "GigaChat:latest"
    gigachat_authorization_key: str  # Base64 key from кабинет (без префикса Basic)
    gigachat_scope: str = "GIGACHAT_API_PERS"
    timeout_sec: float = 120.0
    token_skew_sec: float = 60.0

    ollama_base: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_timeout_sec: float = 120.0
