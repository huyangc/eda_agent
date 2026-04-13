from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.v1.chat import router as chat_router
from app.api.v1.messages import router as messages_router
from app.api.v1.models import router as models_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(chat_router, prefix="/v1")
api_router.include_router(messages_router, prefix="/v1")
api_router.include_router(models_router, prefix="/v1")
