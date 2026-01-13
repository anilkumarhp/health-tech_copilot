"""
Multi-Agent Orchestrator for Healthcare Copilot with LLM integration.
Coordinates LLM-powered agents using intelligent routing.
"""

import asyncio
from typing import Dict, Any, Optional
from loguru import logger

from agents.base import AgentState
from utils.exceptions import handle_exceptions


class MultiAgentOrchestrator:
    """Orchestrates LLM-powered agents with intelligent routing."""
    
    def __init__(self, policy_interpreter, workflow_planner, exception_handler, llm_service):
        """
        Initialize multi-agent orchestrator with LLM routing.
        
        Args:
            policy_interpreter: Policy interpreter agent
            workflow_planner: Workflow planner agent  
            exception_handler: Exception handler agent
            llm_service: LLM service for intelligent routing
        """
        self.policy_interpreter = policy_interpreter
        self.workflow_planner = workflow_planner
        self.exception_handler = exception_handler
        self.llm_service = llm_service
        
        logger.info("Multi-agent orchestrator initialized with LLM routing")
    
    @handle_exceptions
    async def process_complex_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        multi_step: bool = False
    ) -> Dict[str, Any]:
        """
        Process complex query using LLM-powered agent coordination.
        
        Args:
            query: User query
            context: Additional context
            multi_step: Whether to enable multi-step agent processing
            
        Returns:
            Comprehensive response from coordinated agents
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Starting LLM multi-agent processing: {query[:50]}...")
        
        try:
            # Use LLM to route to appropriate agent
            agent_name = await self.llm_service.route_query(query, context or {})
            
            # Process with selected agent
            state = AgentState(query=query, context=context or {})
            
            if agent_name == "PolicyInterpreter":
                result = await self.policy_interpreter.process(state)
            elif agent_name == "WorkflowPlanner":
                result = await self.workflow_planner.process(state)
            elif agent_name == "ExceptionHandler":
                result = await self.exception_handler.process(state)
            else:
                result = await self.policy_interpreter.process(state)  # Default
            
            # Calculate processing time
            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            # Format response
            response = {
                "query": query,
                "agent_used": agent_name,
                "confidence": result.confidence,
                "reasoning": result.reasoning_log,
                "result": result.result,
                "error": result.error,
                "processing_time_ms": processing_time,
                "llm_routing": True
            }
            
            logger.info(f"LLM multi-agent processing completed in {processing_time}ms")
            return response
            
        except Exception as e:
            logger.error(f"LLM multi-agent orchestration failed: {str(e)}")
            return {
                "query": query,
                "agent_used": "None",
                "error": str(e),
                "processing_time_ms": int((asyncio.get_event_loop().time() - start_time) * 1000)
            }
    
    @handle_exceptions
    async def process_workflow_request(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process workflow planning request directly."""
        state = AgentState(query=query, context=context or {})
        result_state = await self.workflow_planner.process(state)
        
        return {
            "query": query,
            "agent_used": "WorkflowPlanner",
            "confidence": result_state.confidence,
            "reasoning": result_state.reasoning_log,
            "result": result_state.result,
            "error": result_state.error
        }
    
    @handle_exceptions
    async def process_exception_request(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process exception handling request directly."""
        state = AgentState(query=query, context=context or {})
        result_state = await self.exception_handler.process(state)
        
        return {
            "query": query,
            "agent_used": "ExceptionHandler", 
            "confidence": result_state.confidence,
            "reasoning": result_state.reasoning_log,
            "result": result_state.result,
            "error": result_state.error
        }
    
    def get_available_agents(self) -> Dict[str, str]:
        """Get information about available agents."""
        return {
            "policy_interpreter": "LLM-powered policy interpretation",
            "workflow_planner": "LLM-powered workflow planning",
            "exception_handler": "LLM-powered exception handling"
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all agents."""
        return {
            "policy_interpreter": True,
            "workflow_planner": True,
            "exception_handler": True,
            "orchestrator": True,
            "llm_routing": self.llm_service.health_check()
        }