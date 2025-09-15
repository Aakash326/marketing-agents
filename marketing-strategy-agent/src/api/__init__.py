"""
API package for the Marketing Strategy Agent.

This package contains FastAPI route implementations for all
marketing strategy endpoints and API functionality.
"""

from .marketing_routes import router as marketing_router
from .content_routes import router as content_router  
from .analytics_routes import router as analytics_router

__all__ = [
    "marketing_router",
    "content_router",
    "analytics_router"
]