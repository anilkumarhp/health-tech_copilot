"""
Workflow Planner Agent for Healthcare Copilot.
Generates step-by-step workflows using LLM for intelligent planning.
"""

from typing import Dict, List
from loguru import logger

from agents.base import BaseAgent, AgentState
from services.vector_service import VectorStoreService
from services.llm_service import LLMService
from utils.exceptions import handle_exceptions


class WorkflowPlannerAgent(BaseAgent):
    """Agent that creates structured workflows using LLM."""
    
    def __init__(self, vector_service: VectorStoreService, llm_service: LLMService):
        """
        Initialize Workflow Planner Agent.
        
        Args:
            vector_service: Vector store service for document retrieval
            llm_service: LLM service for intelligent workflow generation
        """
        super().__init__(
            name="WorkflowPlanner",
            description="Creates structured step-by-step workflows using AI"
        )
        self.vector_service = vector_service
        self.llm_service = llm_service
    
    @handle_exceptions
    async def process(self, state: AgentState) -> AgentState:
        """
        Process workflow planning request using LLM.
        
        Args:
            state: Current agent state with query
            
        Returns:
            Updated state with workflow plan
        """
        try:
            self.log_reasoning(state, f"Starting LLM-powered workflow planning: {state.query[:50]}...")
            
            # Step 1: Get relevant policies
            policy_docs = await self._get_relevant_policies(state.query)
            self.log_reasoning(state, f"Retrieved {len(policy_docs)} relevant policies")
            
            # Step 2: Generate workflow using LLM
            workflow_plan = await self.llm_service.generate_workflow_plan(
                query=state.query,
                policy_docs=policy_docs,
                context=state.context
            )
            
            # Step 3: Enhance workflow with additional details
            enhanced_workflow = self._enhance_workflow(workflow_plan, policy_docs)
            
            # Step 4: Calculate confidence
            confidence = self._calculate_confidence(policy_docs, workflow_plan)
            self.set_confidence(state, confidence, f"LLM workflow with {len(policy_docs)} policies")
            
            # Update state
            state.result = enhanced_workflow
            
            self.log_reasoning(state, "LLM workflow planning completed successfully")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    async def _get_relevant_policies(self, query: str) -> List[Dict]:
        """Get policies relevant to the workflow."""
        enhanced_query = f"{query} workflow procedure policy healthcare"
        results = self.vector_service.search_similar(enhanced_query, max_results=3)
        return [result for result in results if result["score"] > 0.2]
    
    def _enhance_workflow(self, workflow_plan: Dict, policy_docs: List[Dict]) -> Dict:
        """Enhance LLM-generated workflow with additional details."""
        enhanced = workflow_plan.copy()
        
        # Add policy sources
        enhanced["policy_sources"] = [
            {
                "source": doc["source"],
                "relevance_score": doc["score"]
            }
            for doc in policy_docs
        ]
        
        # Add dependencies between steps
        if "steps" in enhanced:
            enhanced["dependencies"] = self._identify_dependencies(enhanced["steps"])
        
        # Add checkpoints
        enhanced["checkpoints"] = self._add_checkpoints(enhanced.get("steps", []))
        
        return enhanced
    
    def _identify_dependencies(self, steps: List[Dict]) -> List[str]:
        """Identify dependencies between workflow steps."""
        dependencies = []
        
        for i, step in enumerate(steps):
            if i > 0:
                dependencies.append(f"Step {i+1} depends on completion of Step {i}")
        
        return dependencies
    
    def _add_checkpoints(self, steps: List[Dict]) -> List[Dict]:
        """Add quality checkpoints to workflow."""
        checkpoints = []
        
        # Add checkpoint after every 2-3 steps
        for i in range(2, len(steps), 3):
            checkpoints.append({
                "after_step": i + 1,
                "checkpoint_type": "Quality Review",
                "description": f"Review completion of steps 1-{i+1}",
                "reviewer_role": "Supervisor"
            })
        
        return checkpoints
    
    def _calculate_confidence(self, policy_docs: List[Dict], workflow_plan: Dict) -> float:
        """Calculate confidence in the generated workflow."""
        base_confidence = 0.8  # Higher base for LLM-generated content
        
        # Boost confidence based on available policies
        policy_boost = min(0.15, len(policy_docs) * 0.05)
        
        # Boost for detailed workflow
        detail_boost = 0.0
        if workflow_plan.get("steps") and len(workflow_plan["steps"]) > 2:
            detail_boost = 0.05
        
        return min(0.95, base_confidence + policy_boost + detail_boost)