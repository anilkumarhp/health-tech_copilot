"""
Multi-agent API endpoints for Healthcare Copilot Phase 3.
"""

from typing import Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from services.agent_service_llm import AgentService
from utils.exceptions import QueryProcessingError

router = APIRouter(prefix="/api/v1", tags=["multi-agents"])

# Service will be injected
agent_service: AgentService = None


def init_services(agent_svc: AgentService):
    """Initialize services for this router."""
    global agent_service
    agent_service = agent_svc


class WorkflowRequest(BaseModel):
    """Request model for workflow planning."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Workflow planning request")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context")


class ExceptionRequest(BaseModel):
    """Request model for exception handling."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Exception or problem description")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context")


class ComplexQueryRequest(BaseModel):
    """Request model for complex multi-agent queries."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Complex query requiring multiple agents")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context")
    multi_step: bool = Field(False, description="Enable multi-step agent processing")


@router.post("/workflow")
async def plan_workflow(request: WorkflowRequest):
    """
    Generate step-by-step workflow for healthcare operations.
    
    Args:
        request: Workflow planning request
        
    Returns:
        Structured workflow with steps, roles, and timelines
    """
    logger.info(f"Workflow planning request: {request.query[:100]}...")
    
    try:
        result = await agent_service.process_workflow_request(
            query=request.query,
            context=request.context
        )
        
        return result
        
    except QueryProcessingError as e:
        logger.error(f"Workflow planning error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error during workflow planning: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during workflow planning")


@router.post("/exception")
async def handle_exception(request: ExceptionRequest):
    """
    Handle exceptions and edge cases in healthcare operations.
    
    Args:
        request: Exception handling request
        
    Returns:
        Exception handling plan with resolution steps and escalation
    """
    logger.info(f"Exception handling request: {request.query[:100]}...")
    
    try:
        result = await agent_service.process_exception_request(
            query=request.query,
            context=request.context
        )
        
        return result
        
    except QueryProcessingError as e:
        logger.error(f"Exception handling error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error during exception handling: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during exception handling")


@router.post("/complex")
async def process_complex_query(request: ComplexQueryRequest):
    """
    Process complex queries using multi-agent coordination.
    
    Args:
        request: Complex query request
        
    Returns:
        Coordinated response from multiple agents
    """
    logger.info(f"Complex query processing: {request.query[:100]}...")
    
    try:
        result = await agent_service.process_complex_query(
            query=request.query,
            context=request.context,
            multi_step=request.multi_step
        )
        
        return result
        
    except QueryProcessingError as e:
        logger.error(f"Complex query processing error: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error during complex query processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during complex query processing")


@router.get("/agents/all")
async def get_all_agents():
    """
    Get information about all available agents including multi-agent capabilities.
    
    Returns:
        Dictionary with all agent information
    """
    try:
        available_agents = agent_service.get_available_agents()
        health_status = agent_service.health_check()
        
        return {
            "agents": available_agents,
            "health": health_status,
            "total_agents": len(available_agents),
            "multi_agent_enabled": True,
            "capabilities": [
                "Policy Interpretation",
                "Workflow Planning", 
                "Exception Handling",
                "Multi-Agent Coordination"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting agent information: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while getting agent information")