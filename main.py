"""
FastAPI application for Healthcare Copilot.
Main application with basic endpoints and service initialization.
"""

from datetime import datetime

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from loguru import logger

from api.v1 import documents, queries, system, agents, multi_agents
from models.schemas import HealthCheckResponse
from services.document_service import DocumentProcessor
from services.query_service import QueryService
from services.vector_service import VectorStoreService
from services.agent_service_llm import AgentService
from services.auth_service import require_admin
from utils.config import settings
from utils.exceptions import global_exception_handler

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Copilot API",
    description="AI-powered healthcare operations assistant for non-clinical tasks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add global exception handler
app.add_exception_handler(Exception, global_exception_handler)

# Initialize services
logger.info("Initializing Healthcare Copilot services...")

try:
    document_processor = DocumentProcessor()
    vector_service = VectorStoreService()
    query_service = QueryService(vector_service)
    agent_service = AgentService(vector_service)
    
    # Initialize API routers with services
    documents.init_services(document_processor, vector_service)
    queries.init_services(agent_service)
    system.init_services(document_processor, vector_service)
    agents.init_services(agent_service)
    multi_agents.init_services(agent_service)
    
    # Include API routers
    app.include_router(documents.router)
    app.include_router(queries.router)
    app.include_router(system.router)
    app.include_router(agents.router)
    app.include_router(multi_agents.router)
    
    logger.info("All services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Healthcare Copilot Enterprise Edition",
        "version": "5.0.0",
        "status": "running",
        "docs": "/docs",
        "features": [
            "Policy Interpretation",
            "Workflow Planning", 
            "Exception Handling",
            "Multi-Agent System",
            "LLM Integration"
        ]
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify all services are running properly.
    
    Returns:
        HealthCheckResponse: System health status
    """
    logger.debug("Performing health check")
    
    # Check individual components
    components = {
        "document_processor": "healthy",
        "vector_store": "healthy" if vector_service.health_check() else "unhealthy",
        "query_service": "healthy" if query_service.health_check() else "unhealthy",
        "agent_service": "healthy" if all(agent_service.health_check().values()) else "unhealthy"
    }
    
    # Determine overall status
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version="5.0.0",
        services=components
    )


@app.get("/api/v1/admin/users")
async def admin_users(user=require_admin):
    """Admin endpoint for user management."""
    return {"users": [], "message": "Admin access granted"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/cache/stats")
async def cache_stats(user=require_admin):
    """Admin endpoint for cache statistics."""
    return {"status": "active", "cache_size": 0, "hit_rate": 0.0}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Healthcare Copilot API server on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )