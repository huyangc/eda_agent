from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 9000
    log_level: str = "info"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    # RAG service uses its own model for doc generation — default to a fast
    # non-reasoning model regardless of what the agent uses for LLM_MODEL.
    # Override with RAG_LLM_MODEL in .env if needed.
    rag_llm_model: str = "qwen3.5-27b"


settings = Settings()
