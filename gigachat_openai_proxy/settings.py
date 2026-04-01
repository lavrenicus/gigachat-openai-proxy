from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gigachat_oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_api_base: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_model: str = "GigaChat:latest"
    gigachat_authorization_key: str  # Base64 key from кабинет (без префикса Basic)
    gigachat_scope: str = "GIGACHAT_API_PERS"
    verify_ssl: bool = True
    ca_bundle: str | None = None
    timeout_sec: float = 120.0
    token_skew_sec: float = 60.0
    gigachat_proxy_debug: bool = False  # env GIGACHAT_PROXY_DEBUG — лог тел запроса/ответа к GigaChat

def ssl_arg(s: Settings) -> bool | str:
    if not s.verify_ssl:
        return False
    return s.ca_bundle if s.ca_bundle else True
