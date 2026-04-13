from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.openai_compat import ModelObject, ModelsListResponse

router = APIRouter()

_MODELS = [
    ModelObject(id="eda-agent"),
    ModelObject(id="eda-agent-primetime"),
    ModelObject(id="eda-agent-innovus"),
]


@router.get("/models")
async def list_models():
    return JSONResponse(content=ModelsListResponse(data=_MODELS).model_dump())
