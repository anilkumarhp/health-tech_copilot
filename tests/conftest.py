import sys
import os
from pathlib import Path
import pytest
from prometheus_client import CollectorRegistry, REGISTRY
from unittest.mock import Mock, AsyncMock, patch

# Add the parent directory to Python path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Clear Prometheus registry before each test to avoid collisions."""
    # Clear all collectors from the default registry
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass  # Already unregistered
    yield
    # Clean up after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass

@pytest.fixture(autouse=True)
def mock_api_services():
    """Mock services at the API module level."""
    with patch('api.v1.queries.agent_service') as mock_queries_agent, \
         patch('api.v1.multi_agents.agent_service') as mock_multi_agent:
        
        # Mock agent service for queries
        mock_queries_agent.interpret_policy = AsyncMock(return_value={
            "result": {
                "policy_summary": "Test policy summary",
                "direct_answer": "Insurance authorization is required for procedures over $500",
                "requirements": ["Test requirement"],
                "procedures": ["Test procedure"],
                "exceptions": [],
                "compliance_notes": []
            },
            "confidence": 0.8,
            "sources": [
                {
                    "content": "Test policy content",
                    "source": "test_policy.pdf",
                    "score": 0.9,
                    "metadata": {}
                }
            ]
        })
        
        # Mock agent service for multi-agents
        mock_multi_agent.process_workflow_request = AsyncMock(return_value={
            "result": {
                "workflow_type": "patient_admission",
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Test step",
                        "estimated_time": "5 minutes",
                        "responsible_role": "Staff",
                        "requirements": []
                    }
                ],
                "total_duration": "30 minutes",
                "required_roles": ["Staff"],
                "compliance_requirements": []
            },
            "confidence": 0.8
        })
        
        mock_multi_agent.process_exception_request = AsyncMock(return_value={
            "result": {
                "exception_type": "system_outage",
                "severity": "high",
                "immediate_actions": ["Test action"],
                "resolution_steps": [],
                "escalation_path": [],
                "prevention_measures": []
            },
            "confidence": 0.8
        })
        
        yield