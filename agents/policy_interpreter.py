"""
Policy Interpreter Agent for Healthcare Copilot.
Interprets healthcare policies using LLM for intelligent responses.
"""

from typing import Dict, List
from loguru import logger

from agents.base import BaseAgent, AgentState
from services.vector_service import VectorStoreService
from services.llm_service import LLMService
from utils.exceptions import handle_exceptions


class PolicyInterpreterAgent(BaseAgent):
    """Agent that interprets healthcare policies using LLM."""
    
    def __init__(self, vector_service: VectorStoreService, llm_service: LLMService):
        """
        Initialize Policy Interpreter Agent.
        
        Args:
            vector_service: Vector store service for document retrieval
            llm_service: LLM service for intelligent interpretation
        """
        super().__init__(
            name="PolicyInterpreter",
            description="Interprets healthcare policies using AI for structured guidance"
        )
        self.vector_service = vector_service
        self.llm_service = llm_service
    
    @handle_exceptions
    async def process(self, state: AgentState) -> AgentState:
        """
        Process policy interpretation request using LLM.
        
        Args:
            state: Current agent state with query
            
        Returns:
            Updated state with policy interpretation
        """
        try:
            self.log_reasoning(state, f"Starting LLM-powered policy interpretation: {state.query[:50]}...")
            
            # Step 1: Retrieve relevant policies
            policy_docs = await self._retrieve_policies(state.query)
            
            if not policy_docs:
                return self._handle_no_policies(state)
            
            self.log_reasoning(state, f"Retrieved {len(policy_docs)} relevant policy documents")
            
            # Step 2: Use LLM for interpretation
            policy_content = "\n\n".join([doc["content"] for doc in policy_docs])
            interpretation = await self.llm_service.generate_policy_interpretation(
                query=state.query,
                policy_content=policy_content,
                context=state.context
            )
            
            # Step 3: Structure the response
            structured_response = self._structure_response(interpretation, policy_docs)
            
            # Step 4: Calculate confidence
            confidence = self._calculate_confidence(policy_docs, interpretation)
            self.set_confidence(state, confidence, f"LLM interpretation with {len(policy_docs)} policies")
            
            # Update state
            state.result = structured_response
            
            self.log_reasoning(state, "LLM policy interpretation completed successfully")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    async def _retrieve_policies(self, query: str) -> List[Dict]:
        """Retrieve relevant policy documents."""
        enhanced_query = f"{query} policy procedure healthcare"
        results = self.vector_service.search_similar(enhanced_query, max_results=5)
        return [result for result in results if result["score"] > 0.3]
    
    def _structure_response(self, interpretation: Dict, policy_docs: List[Dict]) -> Dict:
        """Structure the final response."""
        return {
            "policy_summary": interpretation.get("direct_answer", ""),
            "requirements": interpretation.get("requirements", []),
            "procedures": interpretation.get("procedures", []),
            "exceptions": interpretation.get("exceptions", []),
            "compliance_notes": interpretation.get("compliance_notes", []),
            "sources": [
                {
                    "source": doc["source"],
                    "relevance_score": doc["score"],
                    "excerpt": " ".join(doc["content"][:200].split()) + "..."
                }
                for doc in policy_docs
            ],
            "recommendations": self._generate_recommendations(interpretation),
            "next_steps": self._generate_next_steps(interpretation)
        }
    
    def _generate_recommendations(self, interpretation: Dict) -> List[str]:
        """Generate recommendations based on LLM interpretation."""
        recommendations = []
        
        if interpretation.get("requirements"):
            recommendations.append("Ensure all listed requirements are met")
        
        if interpretation.get("procedures"):
            recommendations.append("Follow documented procedures in sequence")
        
        if interpretation.get("compliance_notes"):
            recommendations.append("Review compliance requirements carefully")
        
        recommendations.append("Consult supervisor if uncertain about any aspect")
        return recommendations
    
    def _generate_next_steps(self, interpretation: Dict) -> List[str]:
        """Generate next steps based on LLM interpretation."""
        next_steps = []
        
        if interpretation.get("procedures"):
            next_steps.append("Begin with the first documented procedure")
        
        next_steps.extend([
            "Document all actions for audit trail",
            "Escalate to supervisor if exceptions apply",
            "Ensure compliance documentation is complete"
        ])
        
        return next_steps
    
    def _calculate_confidence(self, policy_docs: List[Dict], interpretation: Dict) -> float:
        """Calculate confidence score."""
        if not policy_docs:
            return 0.1
        
        # Base confidence on document relevance
        avg_relevance = sum(doc["score"] for doc in policy_docs) / len(policy_docs)
        
        # Boost for LLM-generated content
        llm_boost = 0.2 if interpretation.get("direct_answer") else 0.0
        
        # Source count boost
        source_boost = min(0.2, len(policy_docs) * 0.05)
        
        return min(0.95, avg_relevance + llm_boost + source_boost)
    
    def _handle_no_policies(self, state: AgentState) -> AgentState:
        """Handle case when no relevant policies are found."""
        self.log_reasoning(state, "No relevant policies found")
        
        state.result = {
            "policy_summary": "No specific policies found. Please consult supervisor or policy manual.",
            "requirements": [],
            "procedures": [],
            "exceptions": [],
            "compliance_notes": [],
            "sources": [],
            "recommendations": [
                "Consult with supervisor for guidance",
                "Review complete policy documentation",
                "Document query for policy development review"
            ],
            "next_steps": [
                "Escalate to supervisor",
                "Document the query",
                "Seek department guidance"
            ]
        }
        
        self.set_confidence(state, 0.1, "No relevant policies found")
        return state