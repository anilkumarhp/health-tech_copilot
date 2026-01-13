"""
Pydantic models for Healthcare Copilot API requests and responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    
    filename: str = Field(..., description="Name of the uploaded file")
    content_type: str = Field(..., description="MIME type of the file")
    category: Optional[str] = Field(None, description="Document category (e.g., 'policy', 'sop', 'insurance')")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for document classification")
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        """Validate filename is not empty and has valid extension."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        
        valid_extensions = ['.pdf', '.txt', '.docx']
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"File must have one of these extensions: {valid_extensions}")
        
        return v.strip()


class QueryRequest(BaseModel):
    """Request model for policy queries."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask about policies")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the query")
    max_results: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results to return")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class DocumentInfo(BaseModel):
    """Model for document information."""
    
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    upload_date: datetime = Field(..., description="When the document was uploaded")
    size_bytes: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Processing status")


class QueryResult(BaseModel):
    """Model for individual query result."""
    
    content: str = Field(..., description="Relevant content from the document")
    source: str = Field(..., description="Source document filename")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryResponse(BaseModel):
    """Response model for policy queries."""
    
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer based on the documents")
    results: List[QueryResult] = Field(..., description="Supporting evidence from documents")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the answer")
    processing_time_ms: int = Field(..., description="Time taken to process the query in milliseconds")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Name of the uploaded file")
    status: str = Field(..., description="Upload and processing status")
    message: str = Field(..., description="Human-readable status message")
    processing_time_ms: int = Field(..., description="Time taken to process the document in milliseconds")


class PolicyListResponse(BaseModel):
    """Response model for listing policies."""
    
    documents: List[DocumentInfo] = Field(..., description="List of available documents")
    total_count: int = Field(..., description="Total number of documents")
    categories: List[str] = Field(..., description="Available document categories")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="Application version")
    services: Dict[str, str] = Field(..., description="Status of individual services")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: bool = Field(True, description="Indicates this is an error response")
    message: str = Field(..., description="Human-readable error message")
    error_code: str = Field(..., description="Machine-readable error code")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    path: str = Field(..., description="API path where error occurred")


class AgentQueryRequest(BaseModel):
    """Request model for agent-based queries."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    agent: Optional[str] = Field("policy_interpreter", description="Agent to use for processing")


class AgentQueryResponse(BaseModel):
    """Response model for agent-based queries."""
    
    query: str = Field(..., description="Original query")
    agent_used: str = Field(..., description="Agent that processed the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: List[str] = Field(..., description="Agent reasoning steps")
    result: Dict[str, Any] = Field(..., description="Agent result")
    error: Optional[str] = Field(None, description="Error message if any")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")