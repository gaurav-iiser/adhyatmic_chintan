from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    transcription_model: str = "gpt-transcribe"

    cleaning_model: str = "gpt-5-mini"
    cleaner_context_chars: int = 900

    enrichment_model: str = "gpt-5-mini"
    enrichment_context_chars: int = 1200

    rag_chunking_model: str = "gpt-5-mini"
    rag_chunk_min_chars: int = 600
    rag_chunk_target_chars: int = 1600
    rag_chunk_max_chars: int = 2600

    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 64

    answer_model: str = "gpt-5-mini"
    answer_top_k: int = 5

    chunk_seconds: int = 300
    data_dir: Path = Path("./data")
    glossary_path: Path = Path("./glossary/vedanta_terms.txt")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
