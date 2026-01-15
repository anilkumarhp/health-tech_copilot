"""
Audit Logging Service for Healthcare Copilot.
Implements comprehensive audit trails for HIPAA compliance.
"""

import json
from datetime import datetime, UTC
from typing import Any, Dict, Optional
from pathlib import Path
from loguru import logger


class AuditLogger:
    """Service for comprehensive audit logging."""
    
    def __init__(self, audit_log_path: str = "logs/audit.log"):
        """
        Initialize audit logger.
        
        Args:
            audit_log_path: Path to audit log file
        """
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure separate audit logger
        logger.add(
            self.audit_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            rotation="100 MB",
            retention="1 year",
            compression="zip"
        )
        
        self.logger = logger.bind(service="audit")
    
    def log_query(
        self,
        user_id: str,
        query: str,
        agent_used: str,
        response_summary: str,
        confidence: float,
        processing_time_ms: int,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """
        Log user query for audit trail.
        
        Args:
            user_id: User identifier
            query: User query
            agent_used: Agent that processed the query
            response_summary: Summary of response
            confidence: Confidence score
            processing_time_ms: Processing time
            ip_address: User IP address
            session_id: Session identifier
        """
        audit_entry = {
            'event_type': 'QUERY',
            'timestamp': datetime.now(UTC).isoformat(),
            'user_id': user_id,
            'query': query[:200],  # Truncate for privacy
            'agent_used': agent_used,
            'response_summary': response_summary[:100],
            'confidence': confidence,
            'processing_time_ms': processing_time_ms,
            'ip_address': ip_address,
            'session_id': session_id
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_document_access(
        self,
        user_id: str,
        document_id: str,
        document_name: str,
        action: str,
        ip_address: Optional[str] = None
    ) -> None:
        """
        Log document access for compliance.
        
        Args:
            user_id: User identifier
            document_id: Document identifier
            document_name: Document name
            action: Action performed (view, download, upload, delete)
            ip_address: User IP address
        """
        audit_entry = {
            'event_type': 'DOCUMENT_ACCESS',
            'timestamp': datetime.now(UTC).isoformat(),
            'user_id': user_id,
            'document_id': document_id,
            'document_name': document_name,
            'action': action,
            'ip_address': ip_address
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_authentication(
        self,
        user_id: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        failure_reason: Optional[str] = None
    ) -> None:
        """
        Log authentication events.
        
        Args:
            user_id: User identifier
            action: Action (login, logout, token_refresh)
            success: Whether action succeeded
            ip_address: User IP address
            failure_reason: Reason for failure if applicable
        """
        audit_entry = {
            'event_type': 'AUTHENTICATION',
            'timestamp': datetime.now(UTC).isoformat(),
            'user_id': user_id,
            'action': action,
            'success': success,
            'ip_address': ip_address,
            'failure_reason': failure_reason
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        role: Optional[str] = None
    ) -> None:
        """
        Log authorization decisions.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            granted: Whether access was granted
            role: User role
        """
        audit_entry = {
            'event_type': 'AUTHORIZATION',
            'timestamp': datetime.now(UTC).isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'granted': granted,
            'role': role
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_data_modification(
        self,
        user_id: str,
        entity_type: str,
        entity_id: str,
        action: str,
        changes: Dict[str, Any],
        ip_address: Optional[str] = None
    ) -> None:
        """
        Log data modifications for audit trail.
        
        Args:
            user_id: User identifier
            entity_type: Type of entity modified
            entity_id: Entity identifier
            action: Action performed (create, update, delete)
            changes: Dictionary of changes made
            ip_address: User IP address
        """
        audit_entry = {
            'event_type': 'DATA_MODIFICATION',
            'timestamp': datetime.now(UTC).isoformat(),
            'user_id': user_id,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'action': action,
            'changes': changes,
            'ip_address': ip_address
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        additional_data: Optional[Dict] = None
    ) -> None:
        """
        Log security events.
        
        Args:
            event_type: Type of security event
            severity: Severity level (low, medium, high, critical)
            description: Event description
            user_id: User identifier if applicable
            ip_address: IP address if applicable
            additional_data: Additional event data
        """
        audit_entry = {
            'event_type': 'SECURITY_EVENT',
            'timestamp': datetime.now(UTC).isoformat(),
            'security_event_type': event_type,
            'severity': severity,
            'description': description,
            'user_id': user_id,
            'ip_address': ip_address,
            'additional_data': additional_data or {}
        }
        
        self.logger.warning(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        component: str,
        status: str,
        additional_data: Optional[Dict] = None
    ) -> None:
        """
        Log system events.
        
        Args:
            event_type: Type of system event
            description: Event description
            component: System component
            status: Event status
            additional_data: Additional event data
        """
        audit_entry = {
            'event_type': 'SYSTEM_EVENT',
            'timestamp': datetime.now(UTC).isoformat(),
            'system_event_type': event_type,
            'description': description,
            'component': component,
            'status': status,
            'additional_data': additional_data or {}
        }
        
        self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def log_compliance_check(
        self,
        check_type: str,
        passed: bool,
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> None:
        """
        Log compliance checks.
        
        Args:
            check_type: Type of compliance check
            passed: Whether check passed
            details: Check details
            user_id: User identifier if applicable
        """
        audit_entry = {
            'event_type': 'COMPLIANCE_CHECK',
            'timestamp': datetime.now(UTC).isoformat(),
            'check_type': check_type,
            'passed': passed,
            'details': details,
            'user_id': user_id
        }
        
        level = "INFO" if passed else "WARNING"
        self.logger.log(level, f"AUDIT: {json.dumps(audit_entry)}")
