"""
Agent-based API endpoints for Healthcare Copilot.
"""

from typing import Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from services.agent_service_llm import AgentService
from utils.exceptions import QueryProcessingError

router = APIRouter(prefix="/api/v1", tags=["agents"])

# Service will be injected
agent_service: AgentService = None


def init_services(agent_svc: AgentService):
    """Initialize services for this router."""
    global agent_service
    agent_service = agent_svc


class AgentQueryRequest(BaseModel):
    """Request model for agent-based queries."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context")
    agent: Optional[str] = Field("policy_interpreter", description="Agent to use for processing")


class AgentQueryResponse(BaseModel):
    """Response model for agent-based queries."""
    
    query: str = Field(..., description="Original query")
    agent_used: str = Field(..., description="Agent that processed the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: list = Field(..., description="Agent reasoning steps")
    result: Dict = Field(..., description="Agent result")
    error: Optional[str] = Field(None, description="Error message if any")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


@router.post("/interpret", response_model=AgentQueryResponse)
async def interpret_policy(request: AgentQueryRequest):
    """
    Interpret policy using Policy Interpreter agent.
    
    Args:
        request: Agent query request
        
    Returns:
        AgentQueryResponse: Structured policy interpretation
    """
    logger.info(f"Agent policy interpretation: {request.query[:100]}...")
    
    try:
        result = await agent_service.interpret_policy(
            query=request.query,
            context=request.context
        )
        
        return AgentQueryResponse(**result)
        
    except QueryProcessingError as e:
        logger.error(f"Policy interpretation error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error during policy interpretation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during policy interpretation")


@router.post("/validate", response_model=AgentQueryResponse)
async def validate_action(request: AgentQueryRequest):
    """
    Validate proposed action against policies.
    
    Args:
        request: Agent query request with proposed action
        
    Returns:
        AgentQueryResponse: Validation result
    """
    logger.info(f"Agent action validation: {request.query[:100]}...")
    
    try:
        # Enhance query for validation context
        validation_query = f"Is this action allowed according to policy: {request.query}"
        
        result = await agent_service.interpret_policy(
            query=validation_query,
            context={**request.context, "validation_mode": True}
        )
        
        return AgentQueryResponse(**result)
        
    except QueryProcessingError as e:
        logger.error(f"Action validation error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error during action validation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during action validation")


@router.get("/agents/status")
async def get_agent_status():
    """
    Get status of all available agents.
    
    Returns:
        Dictionary with agent information and health status
    """
    try:
        available_agents = agent_service.get_available_agents()
        health_status = agent_service.health_check()
        
        return {
            "agents": available_agents,
            "health": health_status,
            "total_agents": len(available_agents)
        }
        
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while getting agent status")