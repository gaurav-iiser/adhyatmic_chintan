from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    transcription_model: str = "gpt-transcribe"
    cleaning_model: str = "gpt-5-mini"
    cleaner_context_chars: int = 900
    chunk_seconds: int = 300
    data_dir: Path = Path("./data")
    glossary_path: Path = Path("./glossary/vedanta_terms.txt")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
