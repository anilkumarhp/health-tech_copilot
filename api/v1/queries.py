"""
Query processing API endpoints with LLM integration.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from models.schemas import QueryRequest, QueryResponse
from services.agent_service_llm import AgentService
from utils.exceptions import QueryProcessingError

router = APIRouter(prefix="/api/v1", tags=["queries"])

# Service will be injected
agent_service: AgentService = None


def init_services(agent_svc: AgentService):
    """Initialize services for this router."""
    global agent_service
    agent_service = agent_svc


@router.post("/query", response_model=QueryResponse)
async def query_policies(request: QueryRequest):
    """Query policies using LLM-powered agent."""
    logger.info(f"Processing LLM query: {request.query[:100]}...")
    
    try:
        # Process with LLM-powered agent
        result = await agent_service.interpret_policy(
            query=request.query,
            context=request.context
        )
        
        return QueryResponse(
            query=request.query,
            answer=result.get("result", {}).get("direct_answer", "No answer available"),
            results=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            processing_time_ms=100
        )
        
    except QueryProcessingError as e:
        logger.error(f"LLM query processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during LLM query processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing")


@router.get("/suggestions")
async def get_query_suggestions(q: str = ""):
    """Get query suggestions."""
    suggestions = [
        "What is our appointment scheduling policy?",
        "How do we handle insurance authorization?",
        "What are the patient discharge procedures?"
    ]
    
    if q:
        suggestions = [s for s in suggestions if q.lower() in s.lower()]
    
    return {"suggestions": suggestions}