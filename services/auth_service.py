"""
Authentication and Authorization Service for Healthcare Copilot.
Provides JWT-based authentication and role-based access control.
"""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

from utils.config import settings


class AuthService:
    """JWT-based authentication and authorization service."""
    
    def __init__(self):
        """Initialize authentication service."""
        self.pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
        self.security = HTTPBearer()
        
        # Role permissions
        self.role_permissions = {
            "admin": [
                "read:all", "write:all", "delete:all",
                "manage:users", "manage:system", "view:metrics"
            ],
            "healthcare_staff": [
                "read:policies", "read:workflows", "read:exceptions",
                "write:queries", "write:workflows", "write:exceptions"
            ],
            "supervisor": [
                "read:all", "write:all", "approve:workflows",
                "manage:staff", "view:reports"
            ],
            "viewer": [
                "read:policies", "read:workflows"
            ]
        }
        
        logger.info("Authentication service initialized")
    
    def hash_password(self, password: str) -> str:
        """Hash password using argon2."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiry_hours)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.jwt_secret_key, 
            algorithm=settings.jwt_algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(
            to_encode,
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm
        )
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            
            # Check if token is expired
            if datetime.now(timezone.utc) > datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            return payload
            
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get current user from JWT token."""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        return {
            "user_id": user_id,
            "username": payload.get("username"),
            "role": payload.get("role"),
            "permissions": self.get_user_permissions(payload.get("role"))
        }
    
    def get_user_permissions(self, role: str) -> List[str]:
        """Get permissions for user role."""
        return self.role_permissions.get(role, [])
    
    def check_permission(self, user: Dict, required_permission: str) -> bool:
        """Check if user has required permission."""
        user_permissions = user.get("permissions", [])
        
        # Admin has all permissions
        if "write:all" in user_permissions:
            return True
        
        # Check specific permission
        return required_permission in user_permissions
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission."""
        def decorator(user: Dict = Depends(self.get_current_user)):
            if not self.check_permission(user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission}"
                )
            return user
        return decorator
    
    def require_role(self, required_roles: List[str]):
        """Decorator to require specific roles."""
        def decorator(user: Dict = Depends(self.get_current_user)):
            user_role = user.get("role")
            if user_role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {', '.join(required_roles)}"
                )
            return user
        return decorator


# Global auth service instance
auth_service = AuthService()


# Common permission decorators
require_read_policies = Depends(auth_service.require_permission("read:policies"))
require_write_queries = Depends(auth_service.require_permission("write:queries"))
require_write_workflows = Depends(auth_service.require_permission("write:workflows"))
require_admin = Depends(auth_service.require_role(["admin"]))
require_staff = Depends(auth_service.require_role(["healthcare_staff", "supervisor", "admin"]))


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed within rate limit."""
        now = datetime.now(timezone.utc)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if (now - req_time).seconds < window
        ]
        
        # Check if under limit
        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True
        
        return False


# Global rate limiter
rate_limiter = RateLimiter()