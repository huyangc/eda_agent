from typing import Optional

from langchain_core.language_models import BaseChatModel

from app.config import LLMBackend, settings


def get_llm(
    streaming: bool = False,
    json_mode: bool = False,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Return the configured LangChain chat model."""
    temp = temperature if temperature is not None else 0.1

    if settings.llm_backend == LLMBackend.OPENAI:
        from langchain_openai import ChatOpenAI

        kwargs: dict = dict(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            streaming=streaming,
            temperature=temp,
        )
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    if settings.llm_backend == LLMBackend.BEDROCK:
        from langchain_aws import ChatBedrock  # type: ignore[import]

        return ChatBedrock(
            model_id=settings.bedrock_model_id,
            region_name=settings.bedrock_region,
            streaming=streaming,
        )

    if settings.llm_backend == LLMBackend.OLLAMA:
        from langchain_ollama import ChatOllama  # type: ignore[import]

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"Unknown LLM backend: {settings.llm_backend}")
