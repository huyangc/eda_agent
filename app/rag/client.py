import httpx

from app.config import settings


class RAGClient:
    """Async HTTP client wrapping the external RAG service.

    Expected service contract:
        POST /retrieve
        Request:  { "query": str, "namespace": str, "top_k": int }
        Response: { "docs": [{"content": str, "metadata": dict, "score": float}] }
    """

    _instance: "RAGClient | None" = None

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.rag_service_url,
            timeout=15.0,
        )

    @classmethod
    def get_instance(cls) -> "RAGClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Call the external RAG service and return a list of doc dicts."""
        payload = {
            "query": query,
            "namespace": namespace,
            "top_k": top_k if top_k is not None else settings.rag_top_k,
        }
        try:
            resp = await self._client.post("/retrieve", json=payload)
            resp.raise_for_status()
            return resp.json().get("docs", [])
        except httpx.HTTPError as exc:
            # Log and return empty — agent will still generate a response
            # without retrieved context rather than crashing.
            import logging
            logging.getLogger(__name__).warning(
                "RAG service call failed: %s", exc
            )
            return []

    async def aclose(self) -> None:
        await self._client.aclose()
