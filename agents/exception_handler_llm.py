"""
Exception Handler Agent for Healthcare Copilot.
Handles exceptions and edge cases using LLM for intelligent problem-solving.
"""

from typing import Dict, List
from loguru import logger

from agents.base import BaseAgent, AgentState
from services.vector_service import VectorStoreService
from services.llm_service import LLMService
from utils.exceptions import handle_exceptions


class ExceptionHandlerAgent(BaseAgent):
    """Agent that handles exceptions and edge cases using LLM."""
    
    def __init__(self, vector_service: VectorStoreService, llm_service: LLMService):
        """
        Initialize Exception Handler Agent.
        
        Args:
            vector_service: Vector store service for document retrieval
            llm_service: LLM service for intelligent exception handling
        """
        super().__init__(
            name="ExceptionHandler",
            description="Handles exceptions and edge cases using AI-powered problem solving"
        )
        self.vector_service = vector_service
        self.llm_service = llm_service
    
    @handle_exceptions
    async def process(self, state: AgentState) -> AgentState:
        """
        Process exception handling request using LLM.
        
        Args:
            state: Current agent state with query
            
        Returns:
            Updated state with exception handling plan
        """
        try:
            self.log_reasoning(state, f"Starting LLM-powered exception handling: {state.query[:50]}...")
            
            # Step 1: Get relevant policies for exception handling
            policy_docs = await self._get_relevant_policies(state.query)
            self.log_reasoning(state, f"Retrieved {len(policy_docs)} relevant policies")
            
            # Step 2: Generate exception handling plan using LLM
            exception_plan = await self.llm_service.generate_exception_handling(
                query=state.query,
                policy_docs=policy_docs,
                context=state.context
            )
            
            # Step 3: Enhance plan with additional details
            enhanced_plan = self._enhance_exception_plan(exception_plan, policy_docs)
            
            # Step 4: Calculate confidence
            confidence = self._calculate_confidence(policy_docs, exception_plan)
            self.set_confidence(state, confidence, f"LLM exception handling with {len(policy_docs)} policies")
            
            # Update state
            state.result = enhanced_plan
            
            self.log_reasoning(state, "LLM exception handling completed successfully")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    async def _get_relevant_policies(self, query: str) -> List[Dict]:
        """Get policies relevant to exception handling."""
        enhanced_query = f"{query} exception handling emergency procedure policy"
        results = self.vector_service.search_similar(enhanced_query, max_results=3)
        return [result for result in results if result["score"] > 0.2]
    
    def _enhance_exception_plan(self, exception_plan: Dict, policy_docs: List[Dict]) -> Dict:
        """Enhance LLM-generated exception plan with additional details."""
        enhanced = exception_plan.copy()
        
        # Add policy sources
        enhanced["policy_sources"] = [
            {
                "source": doc["source"],
                "relevance_score": doc["score"]
            }
            for doc in policy_docs
        ]
        
        # Add timeline summary
        enhanced["timeline_summary"] = self._create_timeline_summary(exception_plan)
        
        # Add risk assessment
        enhanced["risk_assessment"] = self._assess_risk_level(exception_plan)
        
        return enhanced
    
    def _create_timeline_summary(self, exception_plan: Dict) -> Dict:
        """Create timeline summary from resolution steps."""
        timeline = {
            "immediate": [],
            "short_term": [],
            "long_term": []
        }
        
        for step in exception_plan.get("resolution_steps", []):
            timeline_key = step.get("timeline", "short_term")
            if timeline_key in timeline:
                timeline[timeline_key].append(step.get("action", ""))
        
        return timeline
    
    def _assess_risk_level(self, exception_plan: Dict) -> Dict:
        """Assess risk level based on exception details."""
        severity = exception_plan.get("severity", "medium")
        
        risk_levels = {
            "low": {
                "risk_score": 2,
                "monitoring_required": False,
                "escalation_needed": False
            },
            "medium": {
                "risk_score": 5,
                "monitoring_required": True,
                "escalation_needed": False
            },
            "high": {
                "risk_score": 8,
                "monitoring_required": True,
                "escalation_needed": True
            },
            "critical": {
                "risk_score": 10,
                "monitoring_required": True,
                "escalation_needed": True
            }
        }
        
        return risk_levels.get(severity, risk_levels["medium"])
    
    def _calculate_confidence(self, policy_docs: List[Dict], exception_plan: Dict) -> float:
        """Calculate confidence in the exception handling plan."""
        base_confidence = 0.75  # Base confidence for LLM-generated content
        
        # Boost confidence based on available policies
        policy_boost = min(0.15, len(policy_docs) * 0.05)
        
        # Boost for detailed plan
        detail_boost = 0.0
        if exception_plan.get("resolution_steps") and len(exception_plan["resolution_steps"]) > 1:
            detail_boost = 0.05
        
        # Severity adjustment
        severity = exception_plan.get("severity", "medium")
        severity_adjustment = {
            "low": 0.05,
            "medium": 0.0,
            "high": -0.05,
            "critical": -0.1
        }.get(severity, 0.0)
        
        return min(0.95, base_confidence + policy_boost + detail_boost + severity_adjustment)