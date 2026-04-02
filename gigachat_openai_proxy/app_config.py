from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    debug: bool = False
    verify_ssl: bool = True
    ca_bundle: str | None = None
    use_mincifry_ca: bool = False
    mincifry_ca_path: str = "certs/mincifry.pem"
    mincifry_ca_url: str = "https://ca.gisca.ru/repository/%D0%9C%D0%98%D0%9D%D0%A6%D0%98%D0%A4%D0%A0%D0%AB.cer"
    pipeline_queue_size: int = 0
    pipeline_workers: int = 1
    pipeline_generator_token: str | None = None

