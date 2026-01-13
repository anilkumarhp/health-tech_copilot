"""
Base agent class for Healthcare Copilot agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from loguru import logger

from utils.exceptions import HealthcareCopilotException


class AgentState(BaseModel):
    """Base state model for agents."""
    
    query: str
    context: Dict[str, Any] = {}
    confidence: float = 0.0
    reasoning: List[str] = []
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all Healthcare Copilot agents."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description
        self.logger = logger.bind(agent=name)
        
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """
        Process the agent state and return updated state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    def log_reasoning(self, state: AgentState, step: str) -> None:
        """
        Log reasoning step.
        
        Args:
            state: Current state
            step: Reasoning step description
        """
        state.reasoning.append(step)
        self.logger.debug(f"Reasoning: {step}")
    
    def set_confidence(self, state: AgentState, confidence: float, reason: str) -> None:
        """
        Set confidence score with reasoning.
        
        Args:
            state: Current state
            confidence: Confidence score (0-1)
            reason: Reason for confidence level
        """
        state.confidence = confidence
        self.log_reasoning(state, f"Confidence set to {confidence:.2f}: {reason}")
    
    def handle_error(self, state: AgentState, error: Exception) -> AgentState:
        """
        Handle agent errors.
        
        Args:
            state: Current state
            error: Exception that occurred
            
        Returns:
            Updated state with error information
        """
        error_msg = str(error)
        state.error = error_msg
        state.confidence = 0.0
        
        self.logger.error(f"Agent {self.name} error: {error_msg}")
        self.log_reasoning(state, f"Error occurred: {error_msg}")
        
        return state