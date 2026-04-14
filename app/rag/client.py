import logging

import httpx

from app.config import settings

log = logging.getLogger(__name__)


class RAGClient:
    """Async HTTP client wrapping the external RAG service.

    Expected service contract:
        GET /retrieve_with_rerank?queries=<q1>&queries=<q2>...
        Response: {
            "status": "success",
            "cmd_list": "cmd_a,cmd_b,cmd_c",
            "data": "<content_a><cmd_gap_flag><content_b><cmd_gap_flag><content_c>"
        }
        Returns max 3 ranked commands. cmd_list[i] ↔ data.split("<cmd_gap_flag>")[i].
    """

    _instance: "RAGClient | None" = None

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.rag_service_url,
            timeout=60.0,
        )

    @classmethod
    def get_instance(cls) -> "RAGClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _parse_response(self, body: dict) -> list[dict]:
        if body.get("status") != "success":
            raise ValueError(f"RAG service non-success status: {body.get('status')!r}")
        cmd_list_str = body.get("cmd_list", "")
        data_str = body.get("data", "")
        if not cmd_list_str or not data_str:
            return []
        cmd_names = [c.strip() for c in cmd_list_str.split(",") if c.strip()]
        contents = data_str.split("<cmd_gap_flag>")
        docs = []
        for rank, (cmd, content) in enumerate(zip(cmd_names, contents)):
            docs.append({
                "content": content.strip(),
                "metadata": {"command_name": cmd, "source_type": "retrieval"},
                "score": round(1.0 - rank * 0.1, 1),  # rank 0→1.0, 1→0.9, 2→0.8
            })
        return docs

    async def retrieve_batch(
        self,
        queries: list[str],
        namespace: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """GET /retrieve_with_rerank with multiple queries in one call."""
        if not queries:
            return []
        try:
            resp = await self._client.get(
                "/retrieve_with_rerank",
                params=[("queries", q) for q in queries[:8]],
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())
        except ValueError as exc:
            log.warning("RAG service returned error body: %s", exc)
            return []
        except httpx.HTTPError as exc:
            log.warning("RAG service call failed: %s", exc)
            return []

    async def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Single-query shim — delegates to retrieve_batch()."""
        return await self.retrieve_batch(queries=[query], namespace=namespace)

    async def aclose(self) -> None:
        await self._client.aclose()
