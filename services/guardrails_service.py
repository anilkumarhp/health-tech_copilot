"""
Guardrails Service for Healthcare Copilot.
Implements content safety, PII detection, and response validation.
"""

import re
from typing import Dict, List, Tuple
from loguru import logger


class GuardrailsService:
    """Service for implementing AI safety guardrails."""
    
    # PII/PHI patterns for healthcare
    PII_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'mrn': r'\b(MRN|Medical Record Number)[:\s]*\d{6,10}\b',
        'dob': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    # Prohibited content patterns
    PROHIBITED_PATTERNS = [
        r'\b(password|secret|api[_-]?key|token)\b',
        r'\b(inject|drop\s+table|union\s+select)\b',  # SQL injection
        r'<script[^>]*>.*?</script>',  # XSS
    ]
    
    # Healthcare-specific sensitive terms
    SENSITIVE_TERMS = [
        'diagnosis', 'treatment', 'medication', 'prescription',
        'patient name', 'patient id', 'insurance number'
    ]
    
    def __init__(self):
        """Initialize guardrails service."""
        self.logger = logger.bind(service="guardrails")
    
    def validate_input(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate input for security and compliance.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        
        # Check for PII/PHI
        pii_found = self._detect_pii(text)
        if pii_found:
            violations.extend([f"PII detected: {pii_type}" for pii_type in pii_found])
        
        # Check for prohibited content
        prohibited = self._detect_prohibited_content(text)
        if prohibited:
            violations.extend([f"Prohibited content: {pattern}" for pattern in prohibited])
        
        # Check input length
        if len(text) > 10000:
            violations.append("Input exceeds maximum length (10000 characters)")
        
        # Check for empty input
        if not text.strip():
            violations.append("Input cannot be empty")
        
        is_valid = len(violations) == 0
        
        if not is_valid:
            self.logger.warning(f"Input validation failed: {violations}")
        
        return is_valid, violations
    
    def validate_output(self, text: str, source_docs: List[str]) -> Tuple[bool, Dict]:
        """
        Validate LLM output for safety and accuracy.
        
        Args:
            text: Output text to validate
            source_docs: Source documents used for generation
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'pii_detected': False,
            'hallucination_risk': 'low',
            'faithfulness_score': 1.0,
            'issues': []
        }
        
        # Check for PII in output
        pii_found = self._detect_pii(text)
        if pii_found:
            report['pii_detected'] = True
            report['issues'].append(f"PII in output: {', '.join(pii_found)}")
        
        # Check for hallucination indicators
        hallucination_risk = self._assess_hallucination_risk(text, source_docs)
        report['hallucination_risk'] = hallucination_risk
        
        if hallucination_risk in ['high', 'critical']:
            report['issues'].append(f"High hallucination risk detected")
        
        # Calculate faithfulness score
        faithfulness = self._calculate_faithfulness(text, source_docs)
        report['faithfulness_score'] = faithfulness
        
        if faithfulness < 0.5:
            report['issues'].append(f"Low faithfulness score: {faithfulness:.2f}")
        
        is_valid = len(report['issues']) == 0
        
        if not is_valid:
            self.logger.warning(f"Output validation failed: {report['issues']}")
        
        return is_valid, report
    
    def sanitize_output(self, text: str) -> str:
        """
        Sanitize output by removing/redacting sensitive information.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        sanitized = text
        
        # Redact PII patterns
        for pii_type, pattern in self.PII_PATTERNS.items():
            sanitized = re.sub(pattern, f'[REDACTED_{pii_type.upper()}]', sanitized, flags=re.IGNORECASE)
        
        # Remove prohibited content
        for pattern in self.PROHIBITED_PATTERNS:
            sanitized = re.sub(pattern, '[REMOVED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def check_content_safety(self, text: str) -> Dict:
        """
        Comprehensive content safety check.
        
        Args:
            text: Text to check
            
        Returns:
            Safety report dictionary
        """
        return {
            'is_safe': True,
            'pii_detected': bool(self._detect_pii(text)),
            'prohibited_content': bool(self._detect_prohibited_content(text)),
            'sensitive_terms': self._detect_sensitive_terms(text),
            'risk_level': self._calculate_risk_level(text)
        }
    
    def _detect_pii(self, text: str) -> List[str]:
        """Detect PII/PHI in text."""
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                found.append(pii_type)
        return found
    
    def _detect_prohibited_content(self, text: str) -> List[str]:
        """Detect prohibited content patterns."""
        found = []
        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(pattern)
        return found
    
    def _detect_sensitive_terms(self, text: str) -> List[str]:
        """Detect healthcare-specific sensitive terms."""
        found = []
        text_lower = text.lower()
        for term in self.SENSITIVE_TERMS:
            if term in text_lower:
                found.append(term)
        return found
    
    def _assess_hallucination_risk(self, output: str, source_docs: List[str]) -> str:
        """
        Assess risk of hallucination in output.
        
        Returns: 'low', 'medium', 'high', or 'critical'
        """
        if not source_docs:
            return 'high'  # No source documents = high risk
        
        # Check for absolute statements without source support
        absolute_patterns = [
            r'\b(always|never|all|none|every|must|cannot)\b',
            r'\b(definitely|certainly|absolutely|guaranteed)\b'
        ]
        
        absolute_count = sum(
            len(re.findall(pattern, output, re.IGNORECASE))
            for pattern in absolute_patterns
        )
        
        # Check for specific numbers/dates without source
        specific_patterns = [
            r'\b\d+%\b',  # Percentages
            r'\$\d+',     # Dollar amounts
            r'\b\d{4}\b'  # Years
        ]
        
        specific_count = sum(
            len(re.findall(pattern, output))
            for pattern in specific_patterns
        )
        
        # Simple heuristic
        if absolute_count > 5 or specific_count > 3:
            return 'high'
        elif absolute_count > 2 or specific_count > 1:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_faithfulness(self, output: str, source_docs: List[str]) -> float:
        """
        Calculate faithfulness score (how well output matches sources).
        
        Returns: Score between 0.0 and 1.0
        """
        if not source_docs:
            return 0.0
        
        # Simple word overlap metric
        output_words = set(output.lower().split())
        source_words = set()
        for doc in source_docs:
            source_words.update(doc.lower().split())
        
        if not output_words:
            return 0.0
        
        overlap = len(output_words.intersection(source_words))
        faithfulness = overlap / len(output_words)
        
        return min(faithfulness, 1.0)
    
    def _calculate_risk_level(self, text: str) -> str:
        """Calculate overall risk level."""
        pii_found = self._detect_pii(text)
        prohibited = self._detect_prohibited_content(text)
        sensitive = self._detect_sensitive_terms(text)
        
        if pii_found or prohibited:
            return 'high'
        elif len(sensitive) > 3:
            return 'medium'
        else:
            return 'low'
