import time
import uuid
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    user: Optional[str] = None
    # EDA-specific extension fields — ignored by unaware clients
    eda_tool: Optional[Literal["primetime", "innovus", "auto"]] = Field(
        default="auto",
        description="Force a specific EDA tool namespace. 'auto' = detect from context.",
    )
    output_mode: Optional[Literal["qa", "tcl", "auto"]] = Field(
        default="auto",
        description="Force output mode. 'auto' = detect from user intent.",
    )


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Streaming response (SSE chunks)
# ---------------------------------------------------------------------------

class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "eda-agent"


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
