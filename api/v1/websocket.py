"""
WebSocket API for real-time Healthcare Copilot communication.
"""

import json
import uuid
from typing import Dict, Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from services.agent_service_llm import AgentService
from services.conversation_service import ConversationMemoryService, ConversationTurn

router = APIRouter()

# Services will be injected
agent_service: AgentService = None
conversation_service: ConversationMemoryService = None

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}
user_sessions: Dict[str, str] = {}  # websocket_id -> session_id


def init_services(agent_svc: AgentService, conv_svc: ConversationMemoryService):
    """Initialize services for WebSocket router."""
    global agent_service, conversation_service
    agent_service = agent_svc
    conversation_service = conv_svc


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_message(self, connection_id: str, message: dict):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections."""
        for websocket in self.active_connections.values():
            await websocket.send_text(json.dumps(message))


manager = ConnectionManager()


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time agent communication."""
    connection_id = str(uuid.uuid4())
    
    try:
        await manager.connect(websocket, connection_id)
        
        # Create or get conversation session
        session_id = f"ws_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        conversation_service.create_session(session_id, user_id)
        user_sessions[connection_id] = session_id
        
        # Send welcome message
        await manager.send_message(connection_id, {
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to Healthcare Copilot",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(connection_id, message)
            
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
        if connection_id in user_sessions:
            del user_sessions[connection_id]
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.send_message(connection_id, {
            "type": "error",
            "message": f"Connection error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


async def handle_websocket_message(connection_id: str, message: dict):
    """Handle incoming WebSocket message."""
    try:
        message_type = message.get("type")
        session_id = user_sessions.get(connection_id)
        
        if not session_id:
            await manager.send_message(connection_id, {
                "type": "error",
                "message": "No active session",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        # Send typing indicator
        await manager.send_message(connection_id, {
            "type": "agent_typing",
            "message": "Agent is processing your request...",
            "timestamp": datetime.now().isoformat()
        })
        
        if message_type == "query":
            await handle_query_message(connection_id, session_id, message)
        elif message_type == "workflow_request":
            await handle_workflow_message(connection_id, session_id, message)
        elif message_type == "exception_request":
            await handle_exception_message(connection_id, session_id, message)
        elif message_type == "complex_query":
            await handle_complex_message(connection_id, session_id, message)
        else:
            await manager.send_message(connection_id, {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {str(e)}")
        await manager.send_message(connection_id, {
            "type": "error",
            "message": f"Processing error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })


async def handle_query_message(connection_id: str, session_id: str, message: dict):
    """Handle policy query message."""
    query = message.get("query", "")
    context = message.get("context", {})
    
    # Add conversation context
    conv_context = conversation_service.get_conversation_context(session_id)
    context.update(conv_context)
    
    # Process query
    result = await agent_service.process_query(query, context)
    
    # Save turn to conversation
    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        user_query=query,
        agent_response=result,
        timestamp=datetime.now(),
        agent_type="PolicyInterpreter",
        confidence=result.get("confidence", 0.0),
        context=context
    )
    conversation_service.add_turn(session_id, turn)
    
    # Send response
    await manager.send_message(connection_id, {
        "type": "query_response",
        "result": result,
        "session_context": conv_context,
        "timestamp": datetime.now().isoformat()
    })


async def handle_workflow_message(connection_id: str, session_id: str, message: dict):
    """Handle workflow planning message."""
    query = message.get("query", "")
    context = message.get("context", {})
    
    # Add conversation context
    conv_context = conversation_service.get_conversation_context(session_id)
    context.update(conv_context)
    
    # Process workflow request
    result = await agent_service.process_workflow_request(query, context)
    
    # Save turn
    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        user_query=query,
        agent_response=result,
        timestamp=datetime.now(),
        agent_type="WorkflowPlanner",
        confidence=result.get("confidence", 0.0),
        context=context
    )
    conversation_service.add_turn(session_id, turn)
    
    # Send response
    await manager.send_message(connection_id, {
        "type": "workflow_response",
        "result": result,
        "session_context": conv_context,
        "timestamp": datetime.now().isoformat()
    })


async def handle_exception_message(connection_id: str, session_id: str, message: dict):
    """Handle exception handling message."""
    query = message.get("query", "")
    context = message.get("context", {})
    
    # Add conversation context
    conv_context = conversation_service.get_conversation_context(session_id)
    context.update(conv_context)
    
    # Process exception request
    result = await agent_service.process_exception_request(query, context)
    
    # Save turn
    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        user_query=query,
        agent_response=result,
        timestamp=datetime.now(),
        agent_type="ExceptionHandler",
        confidence=result.get("confidence", 0.0),
        context=context
    )
    conversation_service.add_turn(session_id, turn)
    
    # Send response
    await manager.send_message(connection_id, {
        "type": "exception_response",
        "result": result,
        "session_context": conv_context,
        "timestamp": datetime.now().isoformat()
    })


async def handle_complex_message(connection_id: str, session_id: str, message: dict):
    """Handle complex multi-agent message."""
    query = message.get("query", "")
    context = message.get("context", {})
    multi_step = message.get("multi_step", False)
    
    # Add conversation context
    conv_context = conversation_service.get_conversation_context(session_id)
    context.update(conv_context)
    
    # Process complex query
    result = await agent_service.process_complex_query(query, context, multi_step)
    
    # Save turn
    turn = ConversationTurn(
        turn_id=str(uuid.uuid4()),
        user_query=query,
        agent_response=result,
        timestamp=datetime.now(),
        agent_type="MultiAgent",
        confidence=result.get("confidence", 0.0),
        context=context
    )
    conversation_service.add_turn(session_id, turn)
    
    # Send response
    await manager.send_message(connection_id, {
        "type": "complex_response",
        "result": result,
        "session_context": conv_context,
        "timestamp": datetime.now().isoformat()
    })


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return {
        "active_connections": len(manager.active_connections),
        "active_sessions": len(user_sessions),
        "conversation_stats": conversation_service.get_session_stats()
    }