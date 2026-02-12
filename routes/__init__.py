"""API router aggregation for all application endpoints."""

from fastapi import APIRouter

from routes.health import router as health_router
from routes.scrape import router as scrape_router
from routes.summarize import router as summarize_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(scrape_router, tags=["scrape"])
api_router.include_router(summarize_router, tags=["summarize"])
