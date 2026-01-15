"""
Document processing service for Healthcare Copilot.
Handles file upload, text extraction, and document management.
"""

import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pypdf
from pypdf import PdfReader
from loguru import logger

from utils.config import settings
from utils.exceptions import DocumentProcessingError, handle_exceptions


class DocumentProcessor:
    """Service for processing and managing documents."""
    
    def __init__(self):
        """Initialize document processor with upload directory."""
        self.upload_dir = Path("./data/uploads")
        self.processed_dir = Path("./data/processed")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file types
        self.supported_types = settings.supported_file_types.split(",")
        self.max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        
        logger.info("Document processor initialized")
    
    @handle_exceptions
    def validate_file(self, filename: str, file_size: int, content_type: str) -> None:
        """
        Validate uploaded file meets requirements.
        
        Args:
            filename: Name of the file
            file_size: Size of the file in bytes
            content_type: MIME type of the file
            
        Raises:
            DocumentProcessingError: If file validation fails
        """
        # Check file size
        if file_size > self.max_size_bytes:
            raise DocumentProcessingError(
                f"File size {file_size} bytes exceeds maximum allowed size of {self.max_size_bytes} bytes",
                filename=filename
            )
        
        # Check file extension
        file_ext = Path(filename).suffix.lower().lstrip('.')
        if file_ext not in self.supported_types:
            raise DocumentProcessingError(
                f"File type '{file_ext}' not supported. Supported types: {self.supported_types}",
                filename=filename
            )
        
        logger.debug(f"File validation passed for {filename}")
    
    @handle_exceptions
    def save_file(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            
        Returns:
            str: Unique document ID
            
        Raises:
            DocumentProcessingError: If file saving fails
        """
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Create safe filename with document ID
        file_ext = Path(filename).suffix
        safe_filename = f"{doc_id}_{filename.replace(' ', '_')}"
        file_path = self.upload_dir / safe_filename
        
        try:
            # Save file to disk
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"File saved: {safe_filename} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to save file: {str(e)}",
                filename=filename
            )
    
    @handle_exceptions
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            DocumentProcessingError: If text extraction fails
        """
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
            
            if not text_content:
                raise DocumentProcessingError(
                    "No text content could be extracted from PDF",
                    filename=file_path.name
                )
            
            extracted_text = "\n\n".join(text_content)
            logger.debug(f"Extracted {len(extracted_text)} characters from PDF")
            
            return extracted_text
            
        except Exception as e:
            if isinstance(e, DocumentProcessingError):
                raise
            raise DocumentProcessingError(
                f"PDF text extraction failed: {str(e)}",
                filename=file_path.name
            )
    
    @handle_exceptions
    def extract_text_from_txt(self, file_path: Path) -> str:
        """
        Extract text content from text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            str: File content
            
        Raises:
            DocumentProcessingError: If text extraction fails
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        logger.debug(f"Successfully read text file with {encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            
            raise DocumentProcessingError(
                "Could not decode text file with any supported encoding",
                filename=file_path.name
            )
            
        except Exception as e:
            if isinstance(e, DocumentProcessingError):
                raise
            raise DocumentProcessingError(
                f"Text file reading failed: {str(e)}",
                filename=file_path.name
            )
    
    @handle_exceptions
    def extract_text(self, doc_id: str, filename: str) -> str:
        """
        Extract text content from uploaded document.
        
        Args:
            doc_id: Document ID
            filename: Original filename
            
        Returns:
            str: Extracted text content
            
        Raises:
            DocumentProcessingError: If text extraction fails
        """
        # Find the saved file
        safe_filename = f"{doc_id}_{filename.replace(' ', '_')}"
        file_path = self.upload_dir / safe_filename
        
        if not file_path.exists():
            raise DocumentProcessingError(
                f"Document file not found: {safe_filename}",
                filename=filename
            )
        
        # Extract text based on file type
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            text_content = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.txt']:
            text_content = self.extract_text_from_txt(file_path)
        else:
            raise DocumentProcessingError(
                f"Text extraction not implemented for file type: {file_ext}",
                filename=filename
            )
        
        # Save extracted text
        text_file_path = self.processed_dir / f"{doc_id}_extracted.txt"
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"Text extracted and saved for document {doc_id}")
        return text_content
    
    @handle_exceptions
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """
        Get information about a processed document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict with document information or None if not found
        """
        # This is a simple implementation - in production, you'd use a database
        text_file_path = self.processed_dir / f"{doc_id}_extracted.txt"
        
        if not text_file_path.exists():
            return None
        
        # Get file stats
        stat = text_file_path.stat()
        
        return {
            "id": doc_id,
            "processed_date": datetime.fromtimestamp(stat.st_mtime),
            "size_bytes": stat.st_size,
            "status": "processed"
        }
    
    @handle_exceptions
    def list_documents(self) -> List[Dict]:
        """
        List all processed documents.
        
        Returns:
            List of document information dictionaries
        """
        documents = []
        
        for text_file in self.processed_dir.glob("*_extracted.txt"):
            doc_id = text_file.stem.replace("_extracted", "")
            doc_info = self.get_document_info(doc_id)
            if doc_info:
                documents.append(doc_info)
        
        logger.debug(f"Found {len(documents)} processed documents")
        return documents