from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.config import settings
from app.logger import setup_logging
from app.rag.client import RAGClient

setup_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the RAG HTTP client (creates the underlying httpx.AsyncClient)
    RAGClient.get_instance()
    yield
    # Graceful shutdown — close the HTTP client
    client = RAGClient.get_instance()
    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="EDA Agent",
        description="OpenAI-compatible EDA assistant with transparent RAG",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
