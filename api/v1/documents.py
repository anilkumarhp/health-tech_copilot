"""
Document management API endpoints.
"""

import time
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from models.schemas import DocumentUploadResponse, PolicyListResponse
from services.document_service import DocumentProcessor
from services.vector_service import VectorStoreService
from utils.exceptions import DocumentProcessingError

router = APIRouter(prefix="/api/v1", tags=["documents"])

# Services will be injected
document_processor: DocumentProcessor = None
vector_service: VectorStoreService = None


def init_services(doc_proc: DocumentProcessor, vec_svc: VectorStoreService):
    """Initialize services for this router."""
    global document_processor, vector_service
    document_processor = doc_proc
    vector_service = vec_svc


@router.post("/ingest", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(None),
    tags: str = Form("")
):
    """Upload and process a policy document."""
    start_time = time.time()
    
    logger.info(f"Processing document upload: {file.filename}")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file
        document_processor.validate_file(
            filename=file.filename,
            file_size=len(file_content),
            content_type=file.content_type
        )
        
        # Save file and get document ID
        doc_id = document_processor.save_file(file_content, file.filename)
        
        # Extract text content
        text_content = document_processor.extract_text(doc_id, file.filename)
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Add to vector store
        metadata = {
            "category": category,
            "tags": ",".join(tag_list) if tag_list else "",  # Convert list to string
            "upload_date": datetime.now().isoformat()
        }
        
        chunks_added = vector_service.add_document(
            text=text_content,
            doc_id=doc_id,
            filename=file.filename,
            metadata=metadata
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Document {doc_id} processed successfully in {processing_time}ms")
        
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status="processed",
            message=f"Document processed successfully. Added {chunks_added} text chunks to knowledge base.",
            processing_time_ms=processing_time
        )
        
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error during document upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during document processing")


@router.get("/policies", response_model=PolicyListResponse)
async def list_policies():
    """List all available policy documents."""
    logger.debug("Listing available policies")
    
    try:
        # Get vector store statistics and documents
        stats = vector_service.get_collection_stats()
        
        # Get all documents from vector store with metadata
        all_docs = vector_service.collection.get(
            include=["metadatas"]
        )
        
        # Group by document ID to avoid duplicates (since we have chunks)
        doc_map = {}
        categories_set = set()
        
        for i, doc_id in enumerate(all_docs["ids"]):
            metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
            original_doc_id = metadata.get("doc_id")
            
            if original_doc_id and original_doc_id not in doc_map:
                # Parse tags back to list
                tags_str = metadata.get("tags", "")
                tags_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
                
                category = metadata.get("category")
                if category:
                    categories_set.add(category)
                
                doc_map[original_doc_id] = {
                    "id": original_doc_id,
                    "filename": metadata.get("filename", f"Document {original_doc_id[:8]}"),
                    "category": category,
                    "tags": tags_list,
                    "upload_date": metadata.get("upload_date", datetime.now().isoformat()),
                    "size_bytes": metadata.get("char_count", 0),
                    "status": "processed"
                }
        
        # Convert to list
        document_infos = list(doc_map.values())
        
        # Get categories
        categories = list(categories_set) if categories_set else ["policy", "sop", "insurance", "procedure"]
        
        return PolicyListResponse(
            documents=document_infos,
            total_count=len(document_infos),
            categories=categories
        )
        
    except Exception as e:
        logger.error(f"Error listing policies: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while listing policies")