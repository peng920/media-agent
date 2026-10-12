from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZIMG_", env_file=".env", extra="ignore")

    engine_base_url: str = "http://127.0.0.1:8188"
    data_dir: Path = Path("data")
    token_secret: str = "dev-secret"
    seed_users: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "app.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
