"""
Global exception handling for Healthcare Copilot.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
import traceback


class HealthcareCopilotException(Exception):
    """Base exception class for Healthcare Copilot."""
    
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", details: Optional[Dict[str, Any]] = None):
        """
        Initialize Healthcare Copilot exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DocumentProcessingError(HealthcareCopilotException):
    """Exception raised during document processing."""
    
    def __init__(self, message: str, filename: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            details={"filename": filename}
        )


class VectorStoreError(HealthcareCopilotException):
    """Exception raised during vector store operations."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            details={"operation": operation}
        )


class QueryProcessingError(HealthcareCopilotException):
    """Exception raised during query processing."""
    
    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="QUERY_PROCESSING_ERROR",
            details={"query": query}
        )


class ValidationError(HealthcareCopilotException):
    """Exception raised during input validation."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field}
        )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for FastAPI application.
    
    Args:
        request: FastAPI request object
        exc: Exception that was raised
        
    Returns:
        JSONResponse with error details
    """
    # Log the exception with full traceback
    logger.error(f"Unhandled exception in {request.url.path}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Handle custom Healthcare Copilot exceptions
    if isinstance(exc, HealthcareCopilotException):
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "message": exc.message,
                "error_code": exc.error_code,
                "details": exc.details,
                "path": str(request.url.path)
            }
        )
    
    # Handle HTTP exceptions
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "error_code": "HTTP_ERROR",
                "details": {"status_code": exc.status_code},
                "path": str(request.url.path)
            }
        )
    
    # Handle all other exceptions
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error occurred",
            "error_code": "INTERNAL_SERVER_ERROR",
            "details": {"exception_type": type(exc).__name__},
            "path": str(request.url.path)
        }
    )


def handle_exceptions(func):
    """
    Decorator to handle exceptions in service functions.
    
    Args:
        func: Function to wrap with exception handling
        
    Returns:
        Wrapped function with exception handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HealthcareCopilotException:
            # Re-raise custom exceptions
            raise
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HealthcareCopilotException(
                message=f"Error in {func.__name__}: {str(e)}",
                error_code="SERVICE_ERROR",
                details={"function": func.__name__, "exception_type": type(e).__name__}
            )
    return wrapper