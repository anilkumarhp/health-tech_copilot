"""
Evaluation and Guardrails API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from loguru import logger

from services.llm_service import LLMService
from services.auth_service import AuthService

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])
auth_service = AuthService()


@router.get("/metrics")
async def get_evaluation_metrics() -> Dict:
    """
    Get aggregate evaluation metrics.
    """
    try:
        llm_service = LLMService()
        metrics = llm_service.get_evaluation_metrics()
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Failed to get evaluation metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/guardrails/status")
async def get_guardrails_status() -> Dict:
    """
    Get guardrails service status.
    """
    try:
        llm_service = LLMService()
        status = llm_service.get_guardrails_status()
        
        return {
            "status": "success",
            "guardrails": status
        }
    except Exception as e:
        logger.error(f"Failed to get guardrails status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
