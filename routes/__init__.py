"""API router aggregation for all application endpoints."""

from fastapi import APIRouter

from routes.health import router as health_router
from routes.scrap import router as scrap_router
from routes.summarize import router as summarize_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(scrap_router, tags=["scrap"])
api_router.include_router(summarize_router, tags=["summarize"])
