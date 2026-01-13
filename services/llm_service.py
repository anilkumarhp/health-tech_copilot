"""
LLM Service for Healthcare Copilot using Ollama.
Provides LLM integration for intelligent agent responses.
"""

import json
from typing import Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from loguru import logger

from utils.config import settings
from utils.exceptions import QueryProcessingError


class LLMService:
    """Service for LLM operations using Ollama."""
    
    def __init__(self):
        """Initialize LLM service with Ollama."""
        try:
            self.llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                timeout=settings.ollama_timeout
            )
            logger.info(f"LLM service initialized with Ollama model: {settings.ollama_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise QueryProcessingError(f"LLM initialization failed: {str(e)}")
    
    async def generate_policy_interpretation(self, query: str, policy_content: str, context: Dict) -> Dict:
        """Generate policy interpretation using LLM."""
        system_prompt = """You are a healthcare operations policy interpreter. Analyze policy content and provide structured responses.

Return your response as valid JSON with this exact structure:
{
    "direct_answer": "Clear answer to the query",
    "requirements": ["requirement1", "requirement2"],
    "procedures": ["step1", "step2"],
    "exceptions": ["exception1", "exception2"],
    "compliance_notes": ["note1", "note2"]
}"""
        
        user_prompt = f"""Query: {query}

Policy Content:
{policy_content}

Context: {json.dumps(context)}

Analyze the policy content and provide a structured interpretation."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                return self._parse_fallback_response(response.content)
                
        except Exception as e:
            logger.error(f"LLM policy interpretation failed: {str(e)}")
            raise QueryProcessingError(f"Policy interpretation failed: {str(e)}")
    
    async def generate_workflow_plan(self, query: str, policy_docs: List[Dict], context: Dict) -> Dict:
        """Generate workflow plan using LLM."""
        system_prompt = """You are a healthcare workflow planner. Create detailed step-by-step workflows.

Return your response as valid JSON with this exact structure:
{
    "workflow_type": "type_of_workflow",
    "steps": [
        {
            "step_number": 1,
            "description": "Step description",
            "estimated_time": "5-10 minutes",
            "responsible_role": "Role name",
            "requirements": ["req1", "req2"]
        }
    ],
    "total_duration": "30-45 minutes",
    "required_roles": ["role1", "role2"],
    "compliance_requirements": ["req1", "req2"]
}"""
        
        policy_content = "\n\n".join([doc["content"] for doc in policy_docs])
        
        user_prompt = f"""Query: {query}

Relevant Policies:
{policy_content}

Context: {json.dumps(context)}

Create a detailed workflow plan with specific steps, timelines, and role assignments."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                return self._parse_workflow_fallback(response.content)
                
        except Exception as e:
            logger.error(f"LLM workflow planning failed: {str(e)}")
            raise QueryProcessingError(f"Workflow planning failed: {str(e)}")
    
    async def generate_exception_handling(self, query: str, policy_docs: List[Dict], context: Dict) -> Dict:
        """Generate exception handling plan using LLM."""
        system_prompt = """You are a healthcare exception handler. Provide solutions for problems and edge cases.

Return your response as valid JSON with this exact structure:
{
    "exception_type": "type_of_exception",
    "severity": "low|medium|high|critical",
    "immediate_actions": ["action1", "action2"],
    "resolution_steps": [
        {
            "step": 1,
            "action": "Action description",
            "responsible": "Role",
            "timeline": "immediate|1 hour|4 hours|24 hours"
        }
    ],
    "escalation_path": ["level1", "level2"],
    "prevention_measures": ["measure1", "measure2"]
}"""
        
        policy_content = "\n\n".join([doc["content"] for doc in policy_docs])
        
        user_prompt = f"""Exception/Problem: {query}

Relevant Policies:
{policy_content}

Context: {json.dumps(context)}

Provide a comprehensive exception handling plan with immediate actions and resolution steps."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                return self._parse_exception_fallback(response.content)
                
        except Exception as e:
            logger.error(f"LLM exception handling failed: {str(e)}")
            raise QueryProcessingError(f"Exception handling failed: {str(e)}")
    
    async def route_query(self, query: str, context: Dict) -> str:
        """Route query to appropriate agent using LLM."""
        system_prompt = """You are a query router for healthcare operations. Determine which agent should handle the query.

Available agents:
- PolicyInterpreter: For policy questions, compliance, regulations
- WorkflowPlanner: For creating step-by-step procedures, workflows
- ExceptionHandler: For problems, errors, edge cases, emergencies

Return only the agent name: PolicyInterpreter, WorkflowPlanner, or ExceptionHandler"""
        
        user_prompt = f"""Query: {query}
Context: {json.dumps(context)}

Which agent should handle this query?"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            agent_name = response.content.strip()
            
            # Validate agent name
            valid_agents = ["PolicyInterpreter", "WorkflowPlanner", "ExceptionHandler"]
            if agent_name in valid_agents:
                return agent_name
            else:
                # Fallback to keyword-based routing
                return self._fallback_routing(query)
                
        except Exception as e:
            logger.error(f"LLM routing failed: {str(e)}")
            return self._fallback_routing(query)
    
    def _parse_fallback_response(self, content: str) -> Dict:
        """Fallback parsing for malformed JSON responses."""
        return {
            "direct_answer": content[:200] + "..." if len(content) > 200 else content,
            "requirements": [],
            "procedures": [],
            "exceptions": [],
            "compliance_notes": []
        }
    
    def _parse_workflow_fallback(self, content: str) -> Dict:
        """Fallback parsing for workflow responses."""
        return {
            "workflow_type": "general_procedure",
            "steps": [
                {
                    "step_number": 1,
                    "description": content[:100] + "..." if len(content) > 100 else content,
                    "estimated_time": "10-15 minutes",
                    "responsible_role": "Healthcare Staff",
                    "requirements": []
                }
            ],
            "total_duration": "15-30 minutes",
            "required_roles": ["Healthcare Staff"],
            "compliance_requirements": []
        }
    
    def _parse_exception_fallback(self, content: str) -> Dict:
        """Fallback parsing for exception responses."""
        return {
            "exception_type": "general_exception",
            "severity": "medium",
            "immediate_actions": [content[:100] + "..." if len(content) > 100 else content],
            "resolution_steps": [
                {
                    "step": 1,
                    "action": "Review the situation and consult supervisor",
                    "responsible": "Healthcare Staff",
                    "timeline": "immediate"
                }
            ],
            "escalation_path": ["Supervisor", "Department Manager"],
            "prevention_measures": []
        }
    
    def _fallback_routing(self, query: str) -> str:
        """Fallback keyword-based routing."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["policy", "regulation", "compliance", "rule"]):
            return "PolicyInterpreter"
        elif any(word in query_lower for word in ["workflow", "procedure", "steps", "process"]):
            return "WorkflowPlanner"
        elif any(word in query_lower for word in ["problem", "error", "exception", "emergency", "issue"]):
            return "ExceptionHandler"
        else:
            return "PolicyInterpreter"  # Default
    
    def health_check(self) -> bool:
        """Check if LLM service is healthy."""
        try:
            # Simple test query
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            return bool(test_response.content)
        except Exception:
            return False