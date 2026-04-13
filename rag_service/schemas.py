from pydantic import BaseModel, Field
from typing import Literal


class RetrieveRequest(BaseModel):
    query: str
    namespace: Literal["primetime", "innovus", "both"]
    top_k: int = Field(default=6, ge=1, le=20)


class DocChunk(BaseModel):
    content: str
    metadata: dict
    score: float


class RetrieveResponse(BaseModel):
    docs: list[DocChunk]
