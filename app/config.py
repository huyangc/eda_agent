from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMBackend(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # LLM
    llm_backend: LLMBackend = LLMBackend.OPENAI
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # External RAG service
    rag_service_url: str = "http://localhost:9000"
    rag_top_k: int = 6
    rag_score_threshold: float = 0.35

    # Agent behaviour
    intent_confidence_threshold: float = 0.7
    max_commands_to_retrieve: int = 5
    # Model used for tool-use decisions in passthrough (should be a fast
    # non-reasoning model — tool routing doesn't need deep reasoning).
    # Overrides llm_model for that specific path only.
    tool_use_model: str = "deepseek-chat"

    # Logging & tracing
    log_dir: str = "logs"


settings = Settings()
