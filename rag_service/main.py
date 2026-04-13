import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from rag_service.config import settings
from rag_service.llm_retriever import generate_mock_docs
from rag_service.schemas import RetrieveRequest, RetrieveResponse

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EDA RAG Service (mock v1)",
    description="Stub RAG service — uses LLM to generate plausible EDA doc chunks.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    logger.debug("retrieve query=%r namespace=%s top_k=%d", req.query, req.namespace, req.top_k)
    docs = await generate_mock_docs(
        query=req.query,
        namespace=req.namespace,
        top_k=req.top_k,
    )
    return RetrieveResponse(docs=docs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "rag_service.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
