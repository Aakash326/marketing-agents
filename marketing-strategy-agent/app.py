"""
Main FastAPI application for the Marketing Strategy Agent.

This is the entry point for the application that sets up the FastAPI server,
includes all routes, and configures middleware.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from src.utils.logging import get_logger
from src.utils.exceptions import (
    MarketingAgentException,
    DatabaseException,
    APIException
)
from src.api.marketing_routes import router as marketing_router
from src.api.content_routes import router as content_router
from src.api.analytics_routes import router as analytics_router
from src.database.tidb_setup import initialize_database

# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Marketing Strategy Agent...")
    
    try:
        # Initialize database
        await initialize_database()
        logger.info("Database initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Marketing Strategy Agent...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered marketing strategy and content generation agent",
    lifespan=lifespan,
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(MarketingAgentException)
async def marketing_agent_exception_handler(request, exc: MarketingAgentException):
    """Handle marketing agent exceptions."""
    logger.error(f"Marketing agent error: {exc.message}", extra={
        "error_code": exc.error_code,
        "details": exc.details,
        "request_path": str(request.url)
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(DatabaseException)
async def database_exception_handler(request, exc: DatabaseException):
    """Handle database exceptions."""
    logger.error(f"Database error: {exc.message}", extra={
        "error_code": exc.error_code,
        "details": exc.details,
        "request_path": str(request.url)
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(APIException)
async def api_exception_handler(request, exc: APIException):
    """Handle API exceptions."""
    logger.error(f"API error: {exc.message}", extra={
        "error_code": exc.error_code,
        "details": exc.details,
        "request_path": str(request.url)
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", extra={
        "exception_type": type(exc).__name__,
        "request_path": str(request.url)
    }, exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "details": str(exc) if settings.debug else None
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    logger.info("Health check requested")
    
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs"
    }


# Include API routers
app.include_router(
    marketing_router,
    prefix="/api/v1",
    tags=["Marketing"]
)

app.include_router(
    content_router,
    prefix="/api/v1",
    tags=["Content"]
)

app.include_router(
    analytics_router,
    prefix="/api/v1",
    tags=["Analytics"]
)


if __name__ == "__main__":
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )