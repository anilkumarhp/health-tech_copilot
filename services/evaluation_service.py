"""
Evaluation Service for Healthcare Copilot.
Implements RAG evaluation metrics and LLM performance monitoring.
"""

import time
from typing import Dict, List
from loguru import logger


class EvaluationService:
    """Service for evaluating RAG and LLM performance."""
    
    def __init__(self):
        """Initialize evaluation service."""
        self.logger = logger.bind(service="evaluation")
        self.metrics_history = []
    
    def evaluate_rag_response(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict],
        ground_truth: str = None
    ) -> Dict:
        """
        Comprehensive RAG evaluation using multiple metrics.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Retrieved documents from vector store
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'answer_relevance': self._calculate_answer_relevance(query, answer),
            'faithfulness': self._calculate_faithfulness(answer, retrieved_docs),
            'context_precision': self._calculate_context_precision(retrieved_docs, answer),
            'context_recall': self._calculate_context_recall(retrieved_docs, answer),
            'retrieval_quality': self._evaluate_retrieval_quality(retrieved_docs),
            'response_completeness': self._check_response_completeness(answer),
            'timestamp': time.time()
        }
        
        # Add ground truth comparison if available
        if ground_truth:
            metrics['accuracy'] = self._calculate_accuracy(answer, ground_truth)
        
        # Calculate overall score
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        # Store metrics for monitoring
        self.metrics_history.append(metrics)
        
        self.logger.info(f"RAG evaluation completed: overall_score={metrics['overall_score']:.2f}")
        
        return metrics
    
    def evaluate_llm_output(
        self,
        prompt: str,
        output: str,
        expected_format: str = None
    ) -> Dict:
        """
        Evaluate LLM output quality.
        
        Args:
            prompt: Input prompt
            output: LLM output
            expected_format: Expected output format (e.g., 'json', 'text')
            
        Returns:
            Evaluation metrics
        """
        metrics = {
            'output_length': len(output),
            'is_coherent': self._check_coherence(output),
            'format_compliance': self._check_format_compliance(output, expected_format),
            'contains_hallucination_markers': self._detect_hallucination_markers(output),
            'confidence_indicators': self._extract_confidence_indicators(output),
            'timestamp': time.time()
        }
        
        self.logger.debug(f"LLM output evaluation: {metrics}")
        
        return metrics
    
    def calculate_latency_metrics(self, start_time: float, end_time: float) -> Dict:
        """
        Calculate latency metrics.
        
        Args:
            start_time: Request start timestamp
            end_time: Request end timestamp
            
        Returns:
            Latency metrics
        """
        latency_ms = (end_time - start_time) * 1000
        
        return {
            'latency_ms': latency_ms,
            'meets_sla': latency_ms < 2000,  # 2 second SLA
            'performance_tier': self._classify_performance(latency_ms)
        }
    
    def get_aggregate_metrics(self, window_size: int = 100) -> Dict:
        """
        Get aggregate metrics over recent evaluations.
        
        Args:
            window_size: Number of recent evaluations to consider
            
        Returns:
            Aggregated metrics
        """
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-window_size:]
        
        return {
            'avg_answer_relevance': self._avg([m['answer_relevance'] for m in recent_metrics]),
            'avg_faithfulness': self._avg([m['faithfulness'] for m in recent_metrics]),
            'avg_context_precision': self._avg([m['context_precision'] for m in recent_metrics]),
            'avg_overall_score': self._avg([m['overall_score'] for m in recent_metrics]),
            'total_evaluations': len(self.metrics_history),
            'window_size': len(recent_metrics)
        }
    
    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Calculate how relevant the answer is to the query.
        Uses simple word overlap and length heuristics.
        """
        if not answer or not query:
            return 0.0
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Word overlap
        overlap = len(query_words.intersection(answer_words))
        relevance = overlap / len(query_words) if query_words else 0.0
        
        # Penalize very short answers
        if len(answer) < 50:
            relevance *= 0.7
        
        return min(relevance, 1.0)
    
    def _calculate_faithfulness(self, answer: str, retrieved_docs: List[Dict]) -> float:
        """
        Calculate faithfulness: how well answer is grounded in retrieved documents.
        """
        if not retrieved_docs or not answer:
            return 0.0
        
        answer_words = set(answer.lower().split())
        doc_words = set()
        
        for doc in retrieved_docs:
            content = doc.get('content', '')
            doc_words.update(content.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(doc_words))
        faithfulness = overlap / len(answer_words)
        
        return min(faithfulness, 1.0)
    
    def _calculate_context_precision(self, retrieved_docs: List[Dict], answer: str) -> float:
        """
        Calculate context precision: how many retrieved docs are actually relevant.
        """
        if not retrieved_docs:
            return 0.0
        
        answer_words = set(answer.lower().split())
        relevant_count = 0
        
        for doc in retrieved_docs:
            content = doc.get('content', '')
            doc_words = set(content.lower().split())
            overlap = len(answer_words.intersection(doc_words))
            
            # Consider doc relevant if >20% word overlap with answer
            if overlap / len(answer_words) > 0.2:
                relevant_count += 1
        
        return relevant_count / len(retrieved_docs)
    
    def _calculate_context_recall(self, retrieved_docs: List[Dict], answer: str) -> float:
        """
        Calculate context recall: how much relevant context was retrieved.
        """
        if not retrieved_docs:
            return 0.0
        
        # Use retrieval scores as proxy for recall
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return min(avg_score, 1.0)
    
    def _evaluate_retrieval_quality(self, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate quality of document retrieval."""
        if not retrieved_docs:
            return {'quality': 'poor', 'score': 0.0}
        
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.8:
            quality = 'excellent'
        elif avg_score > 0.6:
            quality = 'good'
        elif avg_score > 0.4:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'score': avg_score,
            'num_docs': len(retrieved_docs),
            'top_score': max(scores) if scores else 0.0
        }
    
    def _check_response_completeness(self, answer: str) -> Dict:
        """Check if response is complete and well-formed."""
        return {
            'has_content': len(answer.strip()) > 0,
            'min_length_met': len(answer) >= 20,
            'has_structure': any(marker in answer for marker in ['.', ':', '\n']),
            'is_complete': not answer.endswith('...')
        }
    
    def _calculate_accuracy(self, answer: str, ground_truth: str) -> float:
        """Calculate accuracy against ground truth."""
        answer_words = set(answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
        
        overlap = len(answer_words.intersection(truth_words))
        return overlap / len(truth_words)
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate weighted overall score."""
        weights = {
            'answer_relevance': 0.3,
            'faithfulness': 0.3,
            'context_precision': 0.2,
            'context_recall': 0.2
        }
        
        score = sum(
            metrics.get(key, 0.0) * weight
            for key, weight in weights.items()
        )
        
        return min(score, 1.0)
    
    def _check_coherence(self, text: str) -> bool:
        """Check if text is coherent."""
        # Simple heuristics
        if len(text) < 10:
            return False
        
        # Check for repeated words (sign of incoherence)
        words = text.lower().split()
        if len(words) != len(set(words)) and len(set(words)) / len(words) < 0.5:
            return False
        
        return True
    
    def _check_format_compliance(self, output: str, expected_format: str) -> bool:
        """Check if output matches expected format."""
        if not expected_format:
            return True
        
        if expected_format == 'json':
            import json
            try:
                json.loads(output)
                return True
            except:
                return False
        
        return True
    
    def _detect_hallucination_markers(self, text: str) -> List[str]:
        """Detect markers that might indicate hallucination."""
        markers = []
        
        # Absolute statements
        if any(word in text.lower() for word in ['always', 'never', 'definitely', 'certainly']):
            markers.append('absolute_statements')
        
        # Specific numbers without context
        import re
        if re.search(r'\b\d+%\b', text):
            markers.append('specific_percentages')
        
        return markers
    
    def _extract_confidence_indicators(self, text: str) -> Dict:
        """Extract confidence indicators from text."""
        text_lower = text.lower()
        
        return {
            'has_uncertainty': any(word in text_lower for word in ['may', 'might', 'could', 'possibly']),
            'has_confidence': any(word in text_lower for word in ['will', 'must', 'definitely', 'certainly']),
            'has_hedging': any(word in text_lower for word in ['typically', 'generally', 'usually', 'often'])
        }
    
    def _classify_performance(self, latency_ms: float) -> str:
        """Classify performance tier based on latency."""
        if latency_ms < 500:
            return 'excellent'
        elif latency_ms < 1000:
            return 'good'
        elif latency_ms < 2000:
            return 'acceptable'
        else:
            return 'poor'
    
    def _avg(self, values: List[float]) -> float:
        """Calculate average of values."""
        return sum(values) / len(values) if values else 0.0
