"""Enterprise-grade security and secrets management system.

This module provides comprehensive security features including secret management,
API key rotation, encryption, security auditing, and vulnerability scanning for
production-ready evaluation deployments.
"""

import os
import json
import time
import hashlib
import secrets
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from contextlib import contextmanager

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    import hvac  # HashiCorp Vault client

    HAS_VAULT = True
except ImportError:
    HAS_VAULT = False

try:
    import boto3  # AWS SDK

    HAS_AWS = True
except ImportError:
    HAS_AWS = False

from .enhanced_logging import get_logger
from .unified_config import SecurityConfig

logger = get_logger(__name__)


class SecretStoreType(Enum):
    """Types of secret stores supported."""

    LOCAL = "local"
    VAULT = "vault"
    AWS_SECRETS = "aws-secrets"
    AZURE_KEYVAULT = "azure-kv"
    GCP_SECRETS = "gcp-secrets"
    ENVIRONMENT = "environment"


class SecurityLevel(Enum):
    """Security levels for different deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SecretMetadata:
    """Metadata for tracked secrets."""

    secret_id: str
    created_at: datetime
    last_rotated: datetime
    rotation_interval_days: int
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    encrypted: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""

    timestamp: datetime
    event_type: str
    resource: str
    user: Optional[str]
    action: str
    result: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "INFO"


class SecretStore(Protocol):
    """Protocol for secret store implementations."""

    def store_secret(self, key: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store a secret."""
        ...

    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret."""
        ...

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        ...

    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        ...

    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret."""
        ...


class LocalSecretStore:
    """Local encrypted secret store implementation."""

    def __init__(self, storage_path: Path, encryption_key: Optional[bytes] = None):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        if not HAS_CRYPTO:
            logger.warning("Cryptography not available, secrets will be stored in plain text")
            self._cipher = None
        else:
            self._cipher = self._init_encryption(encryption_key)

        self._lock = threading.RLock()
        self._load_secrets()

    def _init_encryption(self, encryption_key: Optional[bytes]) -> Optional[Any]:
        """Initialize encryption cipher."""
        if not HAS_CRYPTO:
            return None

        if encryption_key:
            key = encryption_key
        else:
            # Generate key from password or create new one
            password = os.getenv("OPENEVAL_ENCRYPTION_PASSWORD", "default-dev-password").encode()
            salt = b"openeval-salt"  # In production, use random salt and store securely
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))

        return Fernet(key)

    def _load_secrets(self):
        """Load secrets from storage."""
        self._secrets = {}
        self._metadata = {}

        secrets_file = self.storage_path / "secrets.json"
        metadata_file = self.storage_path / "metadata.json"

        if secrets_file.exists():
            try:
                with open(secrets_file, "r") as f:
                    encrypted_data = json.load(f)

                for key, encrypted_value in encrypted_data.items():
                    if self._cipher:
                        try:
                            decrypted = self._cipher.decrypt(encrypted_value.encode()).decode()
                            self._secrets[key] = decrypted
                        except Exception as e:
                            logger.error(f"Failed to decrypt secret {key}: {e}")
                    else:
                        self._secrets[key] = encrypted_value

            except Exception as e:
                logger.error(f"Failed to load secrets: {e}")

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata_data = json.load(f)

                for key, meta_dict in metadata_data.items():
                    self._metadata[key] = SecretMetadata(
                        secret_id=meta_dict["secret_id"],
                        created_at=datetime.fromisoformat(meta_dict["created_at"]),
                        last_rotated=datetime.fromisoformat(meta_dict["last_rotated"]),
                        rotation_interval_days=meta_dict["rotation_interval_days"],
                        access_count=meta_dict.get("access_count", 0),
                        last_accessed=(
                            datetime.fromisoformat(meta_dict["last_accessed"])
                            if meta_dict.get("last_accessed")
                            else None
                        ),
                        encrypted=meta_dict.get("encrypted", True),
                        tags=meta_dict.get("tags", {}),
                    )
            except Exception as e:
                logger.error(f"Failed to load secret metadata: {e}")

    def _save_secrets(self):
        """Save secrets to storage."""
        secrets_file = self.storage_path / "secrets.json"
        metadata_file = self.storage_path / "metadata.json"

        # Save encrypted secrets
        encrypted_secrets = {}
        for key, value in self._secrets.items():
            if self._cipher:
                encrypted = self._cipher.encrypt(value.encode()).decode()
                encrypted_secrets[key] = encrypted
            else:
                encrypted_secrets[key] = value

        with open(secrets_file, "w") as f:
            json.dump(encrypted_secrets, f, indent=2)

        # Save metadata
        metadata_dict = {}
        for key, metadata in self._metadata.items():
            metadata_dict[key] = {
                "secret_id": metadata.secret_id,
                "created_at": metadata.created_at.isoformat(),
                "last_rotated": metadata.last_rotated.isoformat(),
                "rotation_interval_days": metadata.rotation_interval_days,
                "access_count": metadata.access_count,
                "last_accessed": (
                    metadata.last_accessed.isoformat() if metadata.last_accessed else None
                ),
                "encrypted": metadata.encrypted,
                "tags": metadata.tags,
            }

        with open(metadata_file, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def store_secret(self, key: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store a secret with metadata."""
        with self._lock:
            try:
                self._secrets[key] = value

                # Create or update metadata
                now = datetime.now()
                if key in self._metadata:
                    self._metadata[key].last_rotated = now
                else:
                    self._metadata[key] = SecretMetadata(
                        secret_id=key,
                        created_at=now,
                        last_rotated=now,
                        rotation_interval_days=(
                            metadata.get("rotation_days", 30) if metadata else 30
                        ),
                        tags=metadata.get("tags", {}) if metadata else {},
                    )

                self._save_secrets()
                return True
            except Exception as e:
                logger.error(f"Failed to store secret {key}: {e}")
                return False

    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret and update access metadata."""
        with self._lock:
            if key in self._secrets:
                # Update access metadata
                if key in self._metadata:
                    self._metadata[key].access_count += 1
                    self._metadata[key].last_accessed = datetime.now()
                    self._save_secrets()  # Persist metadata updates

                return self._secrets[key]
            return None

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        with self._lock:
            try:
                if key in self._secrets:
                    del self._secrets[key]
                if key in self._metadata:
                    del self._metadata[key]
                self._save_secrets()
                return True
            except Exception as e:
                logger.error(f"Failed to delete secret {key}: {e}")
                return False

    def list_secrets(self) -> List[str]:
        """List all secret keys."""
        return list(self._secrets.keys())

    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret."""
        return self.store_secret(key, new_value)

    def get_metadata(self, key: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret."""
        return self._metadata.get(key)

    def needs_rotation(self, key: str) -> bool:
        """Check if a secret needs rotation."""
        metadata = self.get_metadata(key)
        if not metadata:
            return False

        days_since_rotation = (datetime.now() - metadata.last_rotated).days
        return days_since_rotation >= metadata.rotation_interval_days


class VaultSecretStore:
    """HashiCorp Vault secret store implementation."""

    def __init__(self, vault_url: str, vault_token: str, mount_point: str = "secret"):
        if not HAS_VAULT:
            raise ImportError("hvac package required for Vault integration")

        self.client = hvac.Client(url=vault_url, token=vault_token)
        self.mount_point = mount_point

        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed")

    def store_secret(self, key: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store secret in Vault."""
        try:
            secret_data = {"data": {"value": value, "metadata": metadata or {}}}

            self.client.secrets.kv.v2.create_or_update_secret(
                path=key, secret=secret_data["data"], mount_point=self.mount_point
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store secret in Vault: {e}")
            return False

    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=key, mount_point=self.mount_point
            )
            return response["data"]["data"]["value"]
        except Exception as e:
            logger.error(f"Failed to retrieve secret from Vault: {e}")
            return None

    def delete_secret(self, key: str) -> bool:
        """Delete secret from Vault."""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=key, mount_point=self.mount_point
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from Vault: {e}")
            return False

    def list_secrets(self) -> List[str]:
        """List secrets from Vault."""
        try:
            response = self.client.secrets.kv.v2.list_secrets(path="", mount_point=self.mount_point)
            return response["data"]["keys"]
        except Exception as e:
            logger.error(f"Failed to list secrets from Vault: {e}")
            return []

    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate secret in Vault."""
        return self.store_secret(key, new_value)


class AWSSecretsStore:
    """AWS Secrets Manager store implementation."""

    def __init__(self, region: str = "us-east-1"):
        if not HAS_AWS:
            raise ImportError("boto3 package required for AWS Secrets Manager")

        self.client = boto3.client("secretsmanager", region_name=region)
        self.region = region

    def store_secret(self, key: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store secret in AWS Secrets Manager."""
        try:
            # Check if secret exists
            try:
                self.client.describe_secret(SecretId=key)
                # Update existing secret
                self.client.update_secret(SecretId=key, SecretString=value)
            except self.client.exceptions.ResourceNotFoundException:
                # Create new secret
                self.client.create_secret(
                    Name=key,
                    SecretString=value,
                    Description=f"OpenEval secret: {metadata.get('description', 'N/A') if metadata else 'N/A'}",
                )

            return True
        except Exception as e:
            logger.error(f"Failed to store secret in AWS: {e}")
            return False

    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=key)
            return response["SecretString"]
        except Exception as e:
            logger.error(f"Failed to retrieve secret from AWS: {e}")
            return None

    def delete_secret(self, key: str) -> bool:
        """Delete secret from AWS Secrets Manager."""
        try:
            self.client.delete_secret(SecretId=key, ForceDeleteWithoutRecovery=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret from AWS: {e}")
            return False

    def list_secrets(self) -> List[str]:
        """List secrets from AWS Secrets Manager."""
        try:
            paginator = self.client.get_paginator("list_secrets")
            secrets = []
            for page in paginator.paginate():
                for secret in page["SecretList"]:
                    secrets.append(secret["Name"])
            return secrets
        except Exception as e:
            logger.error(f"Failed to list secrets from AWS: {e}")
            return []

    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate secret in AWS Secrets Manager."""
        return self.store_secret(key, new_value)


class SecurityAuditor:
    """Security auditing and compliance monitoring."""

    def __init__(self, audit_log_path: Path):
        self.audit_log_path = audit_log_path
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def log_event(
        self,
        event_type: str,
        resource: str,
        action: str,
        result: str,
        user: Optional[str] = None,
        details: Optional[Dict] = None,
        severity: str = "INFO",
    ):
        """Log a security audit event."""
        entry = SecurityAuditEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            resource=resource,
            user=user or "system",
            action=action,
            result=result,
            details=details or {},
            severity=severity,
        )

        self._write_audit_entry(entry)

    def _write_audit_entry(self, entry: SecurityAuditEntry):
        """Write audit entry to log file."""
        with self._lock:
            try:
                log_data = {
                    "timestamp": entry.timestamp.isoformat(),
                    "event_type": entry.event_type,
                    "resource": entry.resource,
                    "user": entry.user,
                    "action": entry.action,
                    "result": entry.result,
                    "details": entry.details,
                    "severity": entry.severity,
                }

                with open(self.audit_log_path, "a") as f:
                    f.write(json.dumps(log_data) + "\n")

            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    def scan_for_violations(self) -> List[Dict[str, Any]]:
        """Scan audit logs for security violations."""
        violations = []

        try:
            if not self.audit_log_path.exists():
                return violations

            with open(self.audit_log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        # Check for suspicious patterns
                        if entry["result"] == "FAILED" and entry["event_type"] == "SECRET_ACCESS":
                            violations.append(
                                {
                                    "type": "FAILED_SECRET_ACCESS",
                                    "timestamp": entry["timestamp"],
                                    "resource": entry["resource"],
                                    "user": entry["user"],
                                    "severity": "HIGH",
                                }
                            )

                        if (
                            entry["event_type"] == "SECRET_ACCESS"
                            and entry.get("details", {}).get("access_count", 0) > 100
                        ):
                            violations.append(
                                {
                                    "type": "EXCESSIVE_SECRET_ACCESS",
                                    "timestamp": entry["timestamp"],
                                    "resource": entry["resource"],
                                    "user": entry["user"],
                                    "severity": "MEDIUM",
                                }
                            )

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Failed to scan audit logs: {e}")

        return violations


class SecurityManager:
    """Main security manager coordinating all security features."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.secret_store = self._init_secret_store()
        self.auditor = self._init_auditor() if config.audit_logging else None
        self._rotation_scheduler = None
        self._vulnerability_scanner = None

        logger.info(f"SecurityManager initialized with {config.secret_store_type} secret store")

    def _init_secret_store(self) -> SecretStore:
        """Initialize the appropriate secret store."""
        store_type = SecretStoreType(self.config.secret_store_type)

        if store_type == SecretStoreType.LOCAL:
            storage_path = Path(self.config.secret_store_config.get("path", ".openeval/secrets"))
            encryption_key = self.config.secret_store_config.get("encryption_key")
            return LocalSecretStore(storage_path, encryption_key)

        elif store_type == SecretStoreType.VAULT:
            if not HAS_VAULT:
                raise ImportError("hvac package required for Vault integration")

            vault_config = self.config.secret_store_config
            return VaultSecretStore(
                vault_url=vault_config["url"],
                vault_token=vault_config["token"],
                mount_point=vault_config.get("mount_point", "secret"),
            )

        elif store_type == SecretStoreType.AWS_SECRETS:
            if not HAS_AWS:
                raise ImportError("boto3 package required for AWS Secrets Manager")

            aws_config = self.config.secret_store_config
            return AWSSecretsStore(region=aws_config.get("region", "us-east-1"))

        else:
            raise ValueError(f"Unsupported secret store type: {store_type}")

    def _init_auditor(self) -> Optional[SecurityAuditor]:
        """Initialize security auditor."""
        if self.config.audit_log_path:
            return SecurityAuditor(Path(self.config.audit_log_path))
        return None

    @contextmanager
    def get_secret(self, key: str, default: Optional[str] = None):
        """Context manager for secure secret access."""
        secret_value = None
        try:
            secret_value = self.secret_store.retrieve_secret(key)
            if secret_value is None:
                secret_value = default

            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_ACCESS",
                    resource=key,
                    action="RETRIEVE",
                    result="SUCCESS" if secret_value else "NOT_FOUND",
                )

            yield secret_value

        except Exception as e:
            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_ACCESS",
                    resource=key,
                    action="RETRIEVE",
                    result="FAILED",
                    details={"error": str(e)},
                    severity="ERROR",
                )
            raise
        finally:
            # Clear secret from memory
            if secret_value:
                secret_value = None

    def store_secret(self, key: str, value: str, metadata: Optional[Dict] = None) -> bool:
        """Store a secret securely."""
        try:
            success = self.secret_store.store_secret(key, value, metadata)

            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_MANAGEMENT",
                    resource=key,
                    action="STORE",
                    result="SUCCESS" if success else "FAILED",
                    details={"metadata": metadata} if metadata else None,
                )

            return success
        except Exception as e:
            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_MANAGEMENT",
                    resource=key,
                    action="STORE",
                    result="FAILED",
                    details={"error": str(e)},
                    severity="ERROR",
                )
            raise

    def rotate_secret(self, key: str, generator: Optional[Callable[[], str]] = None) -> bool:
        """Rotate a secret with optional custom generator."""
        try:
            if generator:
                new_value = generator()
            else:
                # Default: generate secure random string
                new_value = secrets.token_urlsafe(32)

            success = self.secret_store.rotate_secret(key, new_value)

            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_MANAGEMENT",
                    resource=key,
                    action="ROTATE",
                    result="SUCCESS" if success else "FAILED",
                )

            return success
        except Exception as e:
            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_MANAGEMENT",
                    resource=key,
                    action="ROTATE",
                    result="FAILED",
                    details={"error": str(e)},
                    severity="ERROR",
                )
            raise

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        try:
            success = self.secret_store.delete_secret(key)

            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_MANAGEMENT",
                    resource=key,
                    action="DELETE",
                    result="SUCCESS" if success else "FAILED",
                )

            return success
        except Exception as e:
            if self.auditor:
                self.auditor.log_event(
                    event_type="SECRET_MANAGEMENT",
                    resource=key,
                    action="DELETE",
                    result="FAILED",
                    details={"error": str(e)},
                    severity="ERROR",
                )
            raise

    def check_rotations_needed(self) -> List[str]:
        """Check which secrets need rotation."""
        secrets_needing_rotation = []

        if isinstance(self.secret_store, LocalSecretStore):
            for key in self.secret_store.list_secrets():
                if self.secret_store.needs_rotation(key):
                    secrets_needing_rotation.append(key)

        return secrets_needing_rotation

    def auto_rotate_secrets(self):
        """Automatically rotate secrets that are due for rotation."""
        secrets_to_rotate = self.check_rotations_needed()

        for secret_key in secrets_to_rotate:
            try:
                self.rotate_secret(secret_key)
                logger.info(f"Auto-rotated secret: {secret_key}")
            except Exception as e:
                logger.error(f"Failed to auto-rotate secret {secret_key}: {e}")

    def start_rotation_scheduler(self, check_interval_hours: int = 24):
        """Start automatic secret rotation scheduler."""

        def rotation_worker():
            while True:
                try:
                    self.auto_rotate_secrets()
                except Exception as e:
                    logger.error(f"Secret rotation scheduler error: {e}")

                time.sleep(check_interval_hours * 3600)  # Convert to seconds

        self._rotation_scheduler = threading.Thread(target=rotation_worker, daemon=True)
        self._rotation_scheduler.start()
        logger.info(f"Secret rotation scheduler started (interval: {check_interval_hours} hours)")

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        status = {
            "secret_store_type": self.config.secret_store_type,
            "encryption_enabled": self.config.api_key_encryption,
            "audit_logging": self.config.audit_logging,
            "total_secrets": len(self.secret_store.list_secrets()),
            "secrets_needing_rotation": len(self.check_rotations_needed()),
            "vulnerabilities": [],
        }

        if self.auditor:
            violations = self.auditor.scan_for_violations()
            status["security_violations"] = len(violations)
            status["recent_violations"] = violations[:10]  # Last 10

        return status

    def export_security_report(self, output_path: Path):
        """Export comprehensive security report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "security_status": self.get_security_status(),
            "secret_metadata": [],
        }

        # Add secret metadata (without values)
        if isinstance(self.secret_store, LocalSecretStore):
            for key in self.secret_store.list_secrets():
                metadata = self.secret_store.get_metadata(key)
                if metadata:
                    report["secret_metadata"].append(
                        {
                            "key": key,
                            "created_at": metadata.created_at.isoformat(),
                            "last_rotated": metadata.last_rotated.isoformat(),
                            "access_count": metadata.access_count,
                            "needs_rotation": self.secret_store.needs_rotation(key),
                        }
                    )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Security report exported to: {output_path}")


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        if config is None:
            from .unified_config import load_config

            unified_config = load_config()
            config = unified_config.security
        _security_manager = SecurityManager(config)
    return _security_manager


def secure_api_key(api_key: str, service: str) -> str:
    """Store API key securely and return reference."""
    security_manager = get_security_manager()
    key_id = f"api_key_{service}_{hashlib.md5(api_key.encode()).hexdigest()[:8]}"

    metadata = {
        "service": service,
        "description": f"API key for {service}",
        "rotation_days": 30,
        "tags": {"type": "api_key", "service": service},
    }

    security_manager.store_secret(key_id, api_key, metadata)
    return key_id


@contextmanager
def get_api_key(key_reference: str):
    """Get API key securely using reference."""
    security_manager = get_security_manager()
    with security_manager.get_secret(key_reference) as api_key:
        yield api_key
