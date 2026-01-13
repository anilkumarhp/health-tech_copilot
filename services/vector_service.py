"""
Vector store service for Healthcare Copilot.
Handles document embeddings, storage, and similarity search using ChromaDB.
"""

import uuid
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from utils.config import settings
from utils.exceptions import VectorStoreError, handle_exceptions


class VectorStoreService:
    """Service for managing document embeddings and similarity search."""
    
    def __init__(self):
        """Initialize vector store with ChromaDB and embeddings model."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"description": "Healthcare policies and procedures"}
            )
            
            # Use ChromaDB's default embedding function (no external downloads)
            logger.info("Using ChromaDB default embedding function")
            self.embeddings = None  # ChromaDB will handle embeddings internally
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            logger.info("Vector store service initialized successfully")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}")
    
    @handle_exceptions
    def split_text(self, text: str, doc_id: str, filename: str) -> List[Dict]:
        """
        Split document text into chunks for embedding.
        
        Args:
            text: Document text content
            doc_id: Document identifier
            filename: Original filename
            
        Returns:
            List of text chunks with metadata
            
        Raises:
            VectorStoreError: If text splitting fails
        """
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                raise VectorStoreError(
                    "No text chunks generated from document",
                    operation="text_splitting"
                )
            
            # Create chunk documents with metadata
            chunk_docs = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_docs.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "char_count": len(chunk)
                    }
                })
            
            logger.info(f"Split document {doc_id} into {len(chunks)} chunks")
            return chunk_docs
            
        except Exception as e:
            raise VectorStoreError(
                f"Text splitting failed: {str(e)}",
                operation="text_splitting"
            )
    
    @handle_exceptions
    def add_document(self, text: str, doc_id: str, filename: str, metadata: Optional[Dict] = None) -> int:
        """
        Add document to vector store.
        
        Args:
            text: Document text content
            doc_id: Document identifier
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Number of chunks added
            
        Raises:
            VectorStoreError: If document addition fails
        """
        try:
            # Split text into chunks
            chunk_docs = self.split_text(text, doc_id, filename)
            
            # Prepare data for ChromaDB
            ids = [chunk["id"] for chunk in chunk_docs]
            documents = [chunk["text"] for chunk in chunk_docs]
            metadatas = []
            
            for chunk in chunk_docs:
                chunk_metadata = chunk["metadata"].copy()
                if metadata:
                    chunk_metadata.update(metadata)
                metadatas.append(chunk_metadata)
            
            # Add to ChromaDB collection (ChromaDB will generate embeddings automatically)
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added document {doc_id} with {len(chunk_docs)} chunks to vector store")
            return len(chunk_docs)
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to add document to vector store: {str(e)}",
                operation="add_document"
            )
    
    @handle_exceptions
    def search_similar(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of similar document chunks with scores
            
        Raises:
            VectorStoreError: If search fails
        """
        try:
            # Search in ChromaDB (ChromaDB will generate query embedding automatically)
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    # Convert distance to similarity score (0-1, higher is better)
                    distance = results["distances"][0][i]
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": similarity_score,
                        "source": results["metadatas"][0][i].get("filename", "unknown")
                    })
            
            logger.debug(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            raise VectorStoreError(
                f"Similarity search failed: {str(e)}",
                operation="search_similar"
            )
    
    @handle_exceptions
    def remove_document(self, doc_id: str) -> int:
        """
        Remove document and all its chunks from vector store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Number of chunks removed
            
        Raises:
            VectorStoreError: If document removal fails
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            
            if not results["ids"]:
                logger.warning(f"No chunks found for document {doc_id}")
                return 0
            
            # Delete all chunks
            self.collection.delete(ids=results["ids"])
            
            chunks_removed = len(results["ids"])
            logger.info(f"Removed document {doc_id} with {chunks_removed} chunks from vector store")
            
            return chunks_removed
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to remove document from vector store: {str(e)}",
                operation="remove_document"
            )
    
    @handle_exceptions
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
            
        Raises:
            VectorStoreError: If stats retrieval fails
        """
        try:
            # Get collection count
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            # Analyze document sources
            sources = set()
            doc_ids = set()
            
            for metadata in sample_results.get("metadatas", []):
                if metadata:
                    sources.add(metadata.get("filename", "unknown"))
                    doc_ids.add(metadata.get("doc_id", "unknown"))
            
            stats = {
                "total_chunks": count,
                "unique_documents": len(doc_ids),
                "unique_sources": len(sources),
                "collection_name": settings.chroma_collection_name
            }
            
            logger.debug(f"Vector store stats: {stats}")
            return stats
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to get collection stats: {str(e)}",
                operation="get_stats"
            )
    
    @handle_exceptions
    def health_check(self) -> bool:
        """
        Check if vector store is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collection count
            count = self.collection.count()
            logger.debug(f"Vector store health check passed, {count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False