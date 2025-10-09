"""Authentication and authorization framework."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import jwt
import bcrypt


class User:
    """User model for authentication."""

    def __init__(self, username: str, password_hash: str, roles: Set[str] = None):
        self.username = username
        self.password_hash = password_hash
        self.roles = roles or set()
        self.created_at = datetime.utcnow()
        self.last_login = None

    def verify_password(self, password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode(), self.password_hash.encode())

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles


class AuthManager:
    """Authentication and authorization manager."""

    def __init__(self, secret_key: str, token_expiry: timedelta = timedelta(hours=24)):
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict] = {}

    def register_user(self, username: str, password: str, roles: Set[str] = None) -> User:
        """Register a new user."""
        if username in self.users:
            raise ValueError("User already exists")

        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user = User(username, password_hash, roles)
        self.users[username] = user
        return user

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token."""
        user = self.users.get(username)
        if user and user.verify_password(password):
            user.last_login = datetime.utcnow()
            return self._generate_token(user)
        return None

    def authorize(self, token: str, required_roles: List[str] = None) -> Optional[User]:
        """Authorize user from token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            username = payload["sub"]
            user = self.users.get(username)

            if user and required_roles:
                if not any(user.has_role(role) for role in required_roles):
                    return None

            return user
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def _generate_token(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            "sub": user.username,
            "roles": list(user.roles),
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")


class Permission:
    """Permission definition."""

    def __init__(self, resource: str, action: str):
        self.resource = resource
        self.action = action

    def __str__(self):
        return f"{self.resource}:{self.action}"


class RoleBasedAccessControl:
    """RBAC system."""

    def __init__(self):
        self.role_permissions: Dict[str, Set[str]] = {}

    def assign_permission(self, role: str, permission: Permission):
        """Assign permission to role."""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(str(permission))

    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission."""
        for role in user.roles:
            if role in self.role_permissions:
                if str(permission) in self.role_permissions[role]:
                    return True
        return False


__all__ = ["User", "AuthManager", "Permission", "RoleBasedAccessControl"]
