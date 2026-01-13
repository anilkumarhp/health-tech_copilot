"""
Query processing service for Healthcare Copilot.
Handles user queries, retrieves relevant documents, and generates responses.
"""

import time
from typing import Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from loguru import logger

from services.vector_service import VectorStoreService
from utils.exceptions import QueryProcessingError, handle_exceptions


class QueryService:
    """Service for processing user queries and generating responses."""
    
    def __init__(self, vector_service: VectorStoreService):
        """
        Initialize query service with vector store.
        
        Args:
            vector_service: Vector store service instance
        """
        self.vector_service = vector_service
        
        # Initialize LLM (using HuggingFace for free option)
        try:
            # For now, we'll use a simple template-based approach
            # In production, you'd use a proper LLM like OpenAI or HuggingFace
            self.llm = None  # Will implement template-based responses
            
            # Define response template
            self.response_template = PromptTemplate(
                input_variables=["query", "context", "sources"],
                template="""Based on the following healthcare policy documents, please answer the question.

Question: {query}

Relevant Policy Information:
{context}

Sources: {sources}

Answer: Based on the policy documents provided, """
            )
            
            logger.info("Query service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query service: {str(e)}")
            raise QueryProcessingError(f"Query service initialization failed: {str(e)}")
    
    @handle_exceptions
    def process_query(self, query: str, max_results: int = 5, context: Optional[Dict] = None) -> Dict:
        """
        Process user query and generate response.
        
        Args:
            query: User question
            max_results: Maximum number of source documents to retrieve
            context: Additional context for the query
            
        Returns:
            Dictionary with answer, sources, and metadata
            
        Raises:
            QueryProcessingError: If query processing fails
        """
        start_time = time.time()
        
        try:
            # Validate query
            if not query or not query.strip():
                raise QueryProcessingError("Query cannot be empty")
            
            query = query.strip()
            logger.info(f"Processing query: {query[:100]}...")
            
            # Search for relevant documents
            similar_docs = self.vector_service.search_similar(query, max_results)
            
            if not similar_docs:
                return self._generate_no_results_response(query, start_time)
            
            # Generate response based on retrieved documents
            response = self._generate_response(query, similar_docs, context)
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            response["processing_time_ms"] = processing_time
            
            logger.info(f"Query processed successfully in {processing_time}ms")
            return response
            
        except Exception as e:
            if isinstance(e, QueryProcessingError):
                raise
            raise QueryProcessingError(f"Query processing failed: {str(e)}", query=query)
    
    @handle_exceptions
    def _generate_response(self, query: str, similar_docs: List[Dict], context: Optional[Dict] = None) -> Dict:
        """
        Generate response based on retrieved documents.
        
        Args:
            query: Original user query
            similar_docs: List of similar documents from vector search
            context: Additional context
            
        Returns:
            Dictionary with generated response
        """
        # Extract content and sources
        contexts = []
        sources = []
        results = []
        
        for doc in similar_docs:
            # Clean up content by removing extra whitespace and newlines
            cleaned_content = " ".join(doc["content"].split())
            contexts.append(cleaned_content)
            sources.append(doc["source"])
            
            results.append({
                "content": cleaned_content,
                "source": doc["source"],
                "score": doc["score"],
                "metadata": doc.get("metadata", {})
            })
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        unique_sources = list(set(sources))
        
        # Generate answer using template-based approach
        # In production, you'd use an actual LLM here
        answer = self._generate_template_answer(query, combined_context, unique_sources)
        
        # Calculate confidence based on similarity scores
        avg_score = sum(doc["score"] for doc in similar_docs) / len(similar_docs)
        confidence = min(0.95, avg_score)  # Cap at 95% for template-based responses
        
        return {
            "query": query,
            "answer": answer,
            "results": results,
            "confidence": confidence
        }
    
    @handle_exceptions
    def _generate_template_answer(self, query: str, context: str, sources: List[str]) -> str:
        """
        Generate answer using template-based approach.
        
        Args:
            query: User query
            context: Combined context from documents
            sources: List of source documents
            
        Returns:
            Generated answer string
        """
        # Clean up the context by removing extra whitespace and formatting
        cleaned_context = " ".join(context.split())
        
        query_lower = query.lower()
        
        # Check for common healthcare operations queries
        if any(word in query_lower for word in ["appointment", "schedule", "booking"]):
            answer = f"Based on the policy documents, here are the relevant guidelines for appointment scheduling: {cleaned_context[:400]}..."
        elif any(word in query_lower for word in ["insurance", "coverage", "authorization", "pre-auth"]):
            answer = f"According to the insurance policies, the following rules apply: {cleaned_context[:400]}..."
        elif any(word in query_lower for word in ["discharge", "release", "checkout"]):
            answer = f"The discharge procedures specify: {cleaned_context[:400]}..."
        elif any(word in query_lower for word in ["policy", "procedure", "guideline"]):
            answer = f"The relevant policies state: {cleaned_context[:400]}..."
        else:
            answer = f"Based on the available documentation: {cleaned_context[:400]}..."
        
        # Add source information
        if sources:
            source_text = ", ".join(sources[:3])
            if len(sources) > 3:
                source_text += f" and {len(sources) - 3} other documents"
            answer += f" This information is based on: {source_text}"
        
        return answer
    
    @handle_exceptions
    def _generate_no_results_response(self, query: str, start_time: float) -> Dict:
        """
        Generate response when no relevant documents are found.
        
        Args:
            query: Original user query
            start_time: Query start time
            
        Returns:
            Dictionary with no results response
        """
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "query": query,
            "answer": "I couldn't find any relevant information in the available policy documents to answer your question. Please ensure your question relates to healthcare operations, policies, or procedures, or consider uploading additional relevant documents.",
            "results": [],
            "confidence": 0.0,
            "processing_time_ms": processing_time
        }
    
    @handle_exceptions
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input.
        
        Args:
            partial_query: Partial user input
            
        Returns:
            List of suggested queries
        """
        # Simple suggestion system based on common healthcare operations queries
        suggestions = [
            "What are the appointment scheduling policies?",
            "How do I process insurance pre-authorization?",
            "What is the discharge procedure for patients?",
            "What are the requirements for surgery scheduling?",
            "How do I handle insurance claim denials?",
            "What are the visitor policies?",
            "How do I schedule follow-up appointments?",
            "What documentation is required for discharge?",
            "How do I verify insurance coverage?",
            "What are the emergency admission procedures?"
        ]
        
        if not partial_query or len(partial_query) < 2:
            return suggestions[:5]
        
        # Improved matching: prioritize word-start matches over substring matches
        partial_lower = partial_query.lower()
        word_start_matches = []
        substring_matches = []
        
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            words = suggestion_lower.split()
            
            # Check if any word starts with the partial query
            if any(word.startswith(partial_lower) for word in words):
                word_start_matches.append(suggestion)
            # Otherwise check for substring match
            elif partial_lower in suggestion_lower:
                substring_matches.append(suggestion)
        
        # Return word-start matches first, then substring matches
        result = word_start_matches + substring_matches
        return result[:5] if result else []  # Return empty array if no matches
    
    @handle_exceptions
    def health_check(self) -> bool:
        """
        Check if query service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check vector service health
            if not self.vector_service.health_check():
                return False
            
            # Test a simple query
            test_result = self.vector_service.search_similar("test query", max_results=1)
            
            logger.debug("Query service health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Query service health check failed: {str(e)}")
            return False