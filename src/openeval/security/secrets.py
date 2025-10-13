"""Advanced security module with secret stores and audit logging."""

from __future__ import annotations

import json
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading


class SecretStoreType(Enum):
    """Types of secret stores."""

    LOCAL = "local"
    VAULT = "vault"
    AWS_SECRETS = "aws-secrets"
    AZURE_KEYVAULT = "azure-kv"
    GCP_SECRETS = "gcp-secrets"
    ENVIRONMENT = "environment"


class SecurityLevel(Enum):
    """Security levels for operations."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SecretMetadata:
    """Metadata for stored secrets."""

    key: str
    created_at: datetime
    updated_at: datetime
    security_level: SecurityLevel
    tags: Dict[str, str] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    rotation_required: bool = False

    def is_expired(self) -> bool:
        """Check if secret is expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at

    def needs_rotation(self) -> bool:
        """Check if secret needs rotation."""
        return self.rotation_required or self.is_expired()


@dataclass
class SecurityAuditEntry:
    """Audit log entry for security events."""

    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SecretStore:
    """Base class for secret storage."""

    def store(self, key: str, value: str, metadata: SecretMetadata):
        """Store a secret."""
        raise NotImplementedError

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve a secret."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete a secret."""
        raise NotImplementedError

    def list_keys(self) -> List[str]:
        """List all secret keys."""
        raise NotImplementedError

    def rotate_secret(self, key: str) -> bool:
        """Rotate a secret."""
        raise NotImplementedError


class LocalSecretStore(SecretStore):
    """Local file-based secret store."""

    def __init__(self, storage_path: Union[str, Path]):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_secret_path(self, key: str) -> Path:
        """Get path for secret file."""
        # Hash key for filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.storage_path / f"{key_hash}.secret"

    def _get_metadata_path(self, key: str) -> Path:
        """Get path for metadata file."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.storage_path / f"{key_hash}.meta"

    def store(self, key: str, value: str, metadata: SecretMetadata):
        """Store a secret locally."""
        with self._lock:
            secret_path = self._get_secret_path(key)
            meta_path = self._get_metadata_path(key)

            # Encrypt value (simple XOR for demo - use proper encryption in production)
            encrypted_value = self._simple_encrypt(value)

            # Store secret
            with open(secret_path, "w") as f:
                json.dump({"value": encrypted_value}, f)

            # Store metadata
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "key": metadata.key,
                        "created_at": metadata.created_at.isoformat(),
                        "updated_at": metadata.updated_at.isoformat(),
                        "security_level": metadata.security_level.value,
                        "tags": metadata.tags,
                        "expires_at": (
                            metadata.expires_at.isoformat() if metadata.expires_at else None
                        ),
                        "rotation_required": metadata.rotation_required,
                    },
                    f,
                    indent=2,
                )

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve a secret."""
        with self._lock:
            secret_path = self._get_secret_path(key)
            if not secret_path.exists():
                return None

            try:
                with open(secret_path, "r") as f:
                    data = json.load(f)
                    encrypted_value = data["value"]
                    return self._simple_decrypt(encrypted_value)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                return None

    def delete(self, key: str) -> bool:
        """Delete a secret."""
        with self._lock:
            secret_path = self._get_secret_path(key)
            meta_path = self._get_metadata_path(key)

            deleted = False
            if secret_path.exists():
                secret_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
                deleted = True

            return deleted

    def list_keys(self) -> List[str]:
        """List all secret keys."""
        keys = []
        for meta_file in self.storage_path.glob("*.meta"):
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)
                    keys.append(data["key"])
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return keys

    def rotate_secret(self, key: str) -> bool:
        """Rotate a secret by generating a new random value."""
        current_value = self.retrieve(key)
        if current_value is None:
            return False

        # Generate new random secret
        new_value = secrets.token_urlsafe(32)

        # Update metadata
        meta_path = self._get_metadata_path(key)
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    metadata_dict = json.load(f)

                metadata = SecretMetadata(
                    key=metadata_dict["key"],
                    created_at=datetime.fromisoformat(metadata_dict["created_at"]),
                    updated_at=datetime.now(),
                    security_level=SecurityLevel(metadata_dict["security_level"]),
                    tags=metadata_dict.get("tags", {}),
                    expires_at=(
                        datetime.fromisoformat(metadata_dict["expires_at"])
                        if metadata_dict.get("expires_at")
                        else None
                    ),
                    rotation_required=False,
                )

                # Store new value
                self.store(key, new_value, metadata)
                return True
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                return False

        return False

    def _simple_encrypt(self, value: str) -> str:
        """Simple encryption for demo purposes."""
        # Use a fixed key for demo - NEVER do this in production!
        key = b"demo_key_12345"
        encrypted = []
        for i, char in enumerate(value):
            encrypted.append(chr(ord(char) ^ key[i % len(key)]))
        return "".join(encrypted)

    def _simple_decrypt(self, value: str) -> str:
        """Simple decryption for demo purposes."""
        return self._simple_encrypt(value)  # XOR is symmetric


class SecurityManager:
    """Central security manager."""

    def __init__(self, secret_store: SecretStore):
        self.secret_store = secret_store
        self.audit_log: List[SecurityAuditEntry] = []
        self._audit_lock = threading.Lock()

    def store_secret(
        self,
        key: str,
        value: str,
        security_level: SecurityLevel = SecurityLevel.PRODUCTION,
        user_id: Optional[str] = None,
        **metadata_kwargs,
    ) -> bool:
        """Store a secret with audit logging."""
        try:
            metadata = SecretMetadata(
                key=key,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                security_level=security_level,
                **metadata_kwargs,
            )

            self.secret_store.store(key, value, metadata)

            # Audit log
            self._log_event(
                event_type="secret_stored",
                user_id=user_id,
                resource=f"secret:{key}",
                action="store",
                success=True,
                details={"security_level": security_level.value},
            )

            return True
        except Exception as e:
            self._log_event(
                event_type="secret_store_failed",
                user_id=user_id,
                resource=f"secret:{key}",
                action="store",
                success=False,
                details={"error": str(e)},
            )
            return False

    def retrieve_secret(self, key: str, user_id: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret with audit logging."""
        try:
            value = self.secret_store.retrieve(key)

            self._log_event(
                event_type="secret_retrieved",
                user_id=user_id,
                resource=f"secret:{key}",
                action="retrieve",
                success=value is not None,
            )

            return value
        except Exception as e:
            self._log_event(
                event_type="secret_retrieve_failed",
                user_id=user_id,
                resource=f"secret:{key}",
                action="retrieve",
                success=False,
                details={"error": str(e)},
            )
            return None

    def rotate_secret(self, key: str, user_id: Optional[str] = None) -> bool:
        """Rotate a secret."""
        try:
            success = self.secret_store.rotate_secret(key)

            self._log_event(
                event_type="secret_rotated",
                user_id=user_id,
                resource=f"secret:{key}",
                action="rotate",
                success=success,
            )

            return success
        except Exception as e:
            self._log_event(
                event_type="secret_rotate_failed",
                user_id=user_id,
                resource=f"secret:{key}",
                action="rotate",
                success=False,
                details={"error": str(e)},
            )
            return False

    def get_audit_log(
        self, user_id: Optional[str] = None, event_type: Optional[str] = None, limit: int = 100
    ) -> List[SecurityAuditEntry]:
        """Get audit log entries."""
        with self._audit_lock:
            entries = self.audit_log

            if user_id:
                entries = [e for e in entries if e.user_id == user_id]
            if event_type:
                entries = [e for e in entries if e.event_type == event_type]

            return entries[-limit:]

    def _log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log a security event."""
        entry = SecurityAuditEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            details=details or {},
        )

        with self._audit_lock:
            self.audit_log.append(entry)


__all__ = [
    "SecretStoreType",
    "SecurityLevel",
    "SecretMetadata",
    "SecurityAuditEntry",
    "SecretStore",
    "LocalSecretStore",
    "SecurityManager",
]
