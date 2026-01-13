"""
Agent Service for Healthcare Copilot with LLM integration.
Manages all AI agents with Ollama LLM capabilities.
"""

from typing import Dict, List, Optional
from loguru import logger

from agents.policy_interpreter import PolicyInterpreterAgent
from agents.workflow_planner import WorkflowPlannerAgent
from agents.exception_handler_llm import ExceptionHandlerAgent
from agents.multi_agent_orchestrator_llm import MultiAgentOrchestrator
from services.vector_service import VectorStoreService
from services.llm_service import LLMService
from utils.exceptions import QueryProcessingError


class AgentService:
    """Service for managing LLM-powered AI agents."""
    
    def __init__(self, vector_service: VectorStoreService):
        """
        Initialize agent service with LLM integration.
        
        Args:
            vector_service: Vector store service for document retrieval
        """
        self.vector_service = vector_service
        
        # Initialize LLM service
        try:
            self.llm_service = LLMService()
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise QueryProcessingError(f"LLM service initialization failed: {str(e)}")
        
        # Initialize agents with LLM
        self.policy_interpreter = PolicyInterpreterAgent(vector_service, self.llm_service)
        self.workflow_planner = WorkflowPlannerAgent(vector_service, self.llm_service)
        self.exception_handler = ExceptionHandlerAgent(vector_service, self.llm_service)
        
        # Initialize orchestrator with LLM routing
        self.orchestrator = MultiAgentOrchestrator(
            policy_interpreter=self.policy_interpreter,
            workflow_planner=self.workflow_planner,
            exception_handler=self.exception_handler,
            llm_service=self.llm_service
        )
        
        logger.info("Agent service initialized with LLM-powered agents")
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Process policy query using LLM-powered agent."""
        try:
            result = await self.policy_interpreter.process_query(query, context or {})
            return {
                "agent_used": "PolicyInterpreter",
                "result": result.result,
                "confidence": result.confidence,
                "reasoning": result.reasoning_log
            }
        except Exception as e:
            logger.error(f"Policy query processing failed: {str(e)}")
            raise QueryProcessingError(f"Policy query failed: {str(e)}")
    
    async def process_workflow_request(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Process workflow planning using LLM-powered agent."""
        try:
            result = await self.workflow_planner.process_query(query, context or {})
            return {
                "agent_used": "WorkflowPlanner",
                "result": result.result,
                "confidence": result.confidence,
                "reasoning": result.reasoning_log
            }
        except Exception as e:
            logger.error(f"Workflow planning failed: {str(e)}")
            raise QueryProcessingError(f"Workflow planning failed: {str(e)}")
    
    async def process_exception_request(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Process exception handling using LLM-powered agent."""
        try:
            result = await self.exception_handler.process_query(query, context or {})
            return {
                "agent_used": "ExceptionHandler",
                "result": result.result,
                "confidence": result.confidence,
                "reasoning": result.reasoning_log
            }
        except Exception as e:
            logger.error(f"Exception handling failed: {str(e)}")
            raise QueryProcessingError(f"Exception handling failed: {str(e)}")
    
    async def process_complex_query(self, query: str, context: Optional[Dict] = None, multi_step: bool = False) -> Dict:
        """Process complex query using multi-agent orchestration with LLM routing."""
        try:
            result = await self.orchestrator.process_complex_query(query, context or {}, multi_step)
            return result
        except Exception as e:
            logger.error(f"Complex query processing failed: {str(e)}")
            raise QueryProcessingError(f"Complex query failed: {str(e)}")
    
    def get_available_agents(self) -> List[Dict]:
        """Get information about available LLM-powered agents."""
        return [
            {
                "name": "PolicyInterpreter",
                "description": "Interprets healthcare policies using AI",
                "capabilities": ["Policy analysis", "Compliance guidance", "Requirement extraction"],
                "llm_powered": True
            },
            {
                "name": "WorkflowPlanner", 
                "description": "Creates step-by-step workflows using AI",
                "capabilities": ["Workflow generation", "Task sequencing", "Role assignment"],
                "llm_powered": True
            },
            {
                "name": "ExceptionHandler",
                "description": "Handles exceptions and problems using AI",
                "capabilities": ["Problem solving", "Risk assessment", "Escalation planning"],
                "llm_powered": True
            }
        ]
    
    def health_check(self) -> Dict:
        """Check health of all agents and LLM service."""
        return {
            "policy_interpreter": True,
            "workflow_planner": True,
            "exception_handler": True,
            "orchestrator": True,
            "llm_service": self.llm_service.health_check()
        }