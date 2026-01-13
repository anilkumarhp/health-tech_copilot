"""
System monitoring and statistics API endpoints.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from services.document_service import DocumentProcessor
from services.vector_service import VectorStoreService

router = APIRouter(prefix="/api/v1", tags=["system"])

# Services will be injected
document_processor: DocumentProcessor = None
vector_service: VectorStoreService = None


def init_services(doc_proc: DocumentProcessor, vec_svc: VectorStoreService):
    """Initialize services for this router."""
    global document_processor, vector_service
    document_processor = doc_proc
    vector_service = vec_svc


@router.get("/stats")
async def get_system_stats():
    """Get system statistics and metrics."""
    try:
        # Get vector store stats
        vector_stats = vector_service.get_collection_stats()
        
        # Get document processor stats
        documents = document_processor.list_documents()
        
        return {
            "vector_store": vector_stats,
            "documents": {
                "total_processed": len(documents),
                "processing_status": "healthy"
            },
            "system": {
                "uptime": "running",
                "version": "1.0.0"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while getting statistics")