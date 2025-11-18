"""HTTP connection pooling for adapters.

Uses httpx.Client with connection pooling for reusable connections.
"""

from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None


class PooledHTTPClient:
    """HTTP client with connection pooling."""

    _client: Optional[httpx.Client] = None

    @classmethod
    def get_client(cls, timeout: float = 30.0) -> Optional[httpx.Client]:
        """Get or create pooled HTTP client."""
        if httpx is None:
            return None

        if cls._client is None:
            cls._client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )
        return cls._client

    @classmethod
    def close(cls):
        """Close the pooled client."""
        if cls._client is not None:
            cls._client.close()
            cls._client = None


def get_pooled_http_client(timeout: float = 30.0) -> Optional[httpx.Client]:
    """Get a pooled HTTP client for reusing connections."""
    return PooledHTTPClient.get_client(timeout)


def close_pooled_client():
    """Close the pooled HTTP client."""
    PooledHTTPClient.close()


__all__ = ["PooledHTTPClient", "get_pooled_http_client", "close_pooled_client"]
