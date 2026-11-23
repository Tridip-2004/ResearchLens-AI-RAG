from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # Application settings
    app_name: str = "ResearchLens AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # Storage
    data_dir: str = str(BASE_DIR / "data")
    collection_name: str = "researchlens"

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Reranking model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Local language model
    llm_provider: str = "ollama"
    ollama_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"

    # Optional OpenAI configuration
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Fallback settings
    allow_extractive_fallback: bool = True

    # Retrieval
    top_k: int = 5

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()