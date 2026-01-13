"""
Metrics Service for Healthcare Copilot.
Provides Prometheus metrics for enterprise monitoring.
"""

import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger


class MetricsService:
    """Prometheus metrics collection service."""
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        
        # Request metrics
        self.request_count = Counter(
            'healthcare_copilot_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'healthcare_copilot_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Agent metrics
        self.agent_requests = Counter(
            'healthcare_copilot_agent_requests_total',
            'Total agent requests',
            ['agent_type']
        )
        
        self.agent_duration = Histogram(
            'healthcare_copilot_agent_duration_seconds',
            'Agent processing duration',
            ['agent_type']
        )
        
        self.agent_confidence = Histogram(
            'healthcare_copilot_agent_confidence',
            'Agent confidence scores',
            ['agent_type'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # LLM metrics
        self.llm_requests = Counter(
            'healthcare_copilot_llm_requests_total',
            'Total LLM requests',
            ['model', 'status']
        )
        
        self.llm_duration = Histogram(
            'healthcare_copilot_llm_duration_seconds',
            'LLM processing duration',
            ['model']
        )
        
        self.llm_tokens = Counter(
            'healthcare_copilot_llm_tokens_total',
            'Total LLM tokens processed',
            ['model', 'type']
        )
        
        # Cache metrics
        self.cache_requests = Counter(
            'healthcare_copilot_cache_requests_total',
            'Total cache requests',
            ['cache_type', 'result']
        )
        
        # Vector store metrics
        self.vector_searches = Counter(
            'healthcare_copilot_vector_searches_total',
            'Total vector searches'
        )
        
        self.vector_search_duration = Histogram(
            'healthcare_copilot_vector_search_duration_seconds',
            'Vector search duration'
        )
        
        # System metrics
        self.active_sessions = Gauge(
            'healthcare_copilot_active_sessions',
            'Number of active sessions'
        )
        
        self.websocket_connections = Gauge(
            'healthcare_copilot_websocket_connections',
            'Number of active WebSocket connections'
        )
        
        # Application info
        self.app_info = Info(
            'healthcare_copilot_info',
            'Application information'
        )
        
        # Set application info
        self.app_info.info({
            'version': '5.0.0',
            'phase': 'Enterprise',
            'llm_enabled': 'true',
            'cache_enabled': 'true'
        })
        
        logger.info("Metrics service initialized with Prometheus collectors")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_agent_request(self, agent_type: str, duration: float, confidence: float):
        """Record agent processing metrics."""
        self.agent_requests.labels(agent_type=agent_type).inc()
        self.agent_duration.labels(agent_type=agent_type).observe(duration)
        self.agent_confidence.labels(agent_type=agent_type).observe(confidence)
    
    def record_llm_request(self, model: str, duration: float, status: str = "success", 
                          input_tokens: int = 0, output_tokens: int = 0):
        """Record LLM request metrics."""
        self.llm_requests.labels(model=model, status=status).inc()
        self.llm_duration.labels(model=model).observe(duration)
        
        if input_tokens > 0:
            self.llm_tokens.labels(model=model, type="input").inc(input_tokens)
        if output_tokens > 0:
            self.llm_tokens.labels(model=model, type="output").inc(output_tokens)
    
    def record_cache_request(self, cache_type: str, hit: bool):
        """Record cache request metrics."""
        result = "hit" if hit else "miss"
        self.cache_requests.labels(cache_type=cache_type, result=result).inc()
    
    def record_vector_search(self, duration: float):
        """Record vector search metrics."""
        self.vector_searches.inc()
        self.vector_search_duration.observe(duration)
    
    def set_active_sessions(self, count: int):
        """Set active sessions gauge."""
        self.active_sessions.set(count)
    
    def set_websocket_connections(self, count: int):
        """Set WebSocket connections gauge."""
        self.websocket_connections.set(count)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest()
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global metrics instance
metrics = MetricsService()


class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        # Normalize endpoint for metrics
        endpoint = self._normalize_endpoint(path)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                metrics.record_request(method, endpoint, status_code, duration)
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Remove query parameters
        if "?" in path:
            path = path.split("?")[0]
        
        # Normalize common patterns
        if path.startswith("/api/v1/"):
            return path
        elif path == "/":
            return "/"
        elif path == "/health":
            return "/health"
        elif path == "/metrics":
            return "/metrics"
        elif path.startswith("/ws/"):
            return "/ws/{user_id}"
        else:
            return "/other"