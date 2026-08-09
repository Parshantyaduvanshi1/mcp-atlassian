"""In-memory cache for Jira attachments to expose via MCP resources."""

import hashlib
import importlib
import json
import logging
import os
import re
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mcp-jira")

# Download tokens are URL-safe (token_urlsafe); validate before using on disk.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 string into an aware datetime, or None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class DownloadTokenStore(ABC):
    """Pluggable store mapping a download token to self-contained file bytes.

    Tokens are self-contained: each carries its own content and expiry, so a
    token minted on one instance can be resolved by any other instance sharing
    the store. Point the ``filesystem`` backend at a shared volume (EFS, Azure
    Files, NFS, ...) for stateless multi-instance downloads; the default
    in-memory store keeps single-instance behavior with no extra dependencies.
    """

    @abstractmethod
    def create(
        self,
        issue_key: str,
        filename: str,
        mime_type: str,
        content: bytes,
        expires_at: datetime,
    ) -> str:
        """Persist a token entry and return the opaque token."""

    @abstractmethod
    def get(self, token: str) -> dict[str, Any] | None:
        """Resolve a token to its entry, or None if unknown / expired."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all tokens."""

    @abstractmethod
    def sweep_expired(self) -> int:
        """Delete every expired token entry; return the number removed."""


class MemoryDownloadTokenStore(DownloadTokenStore):
    """Process-local, in-memory download token store (single instance)."""

    def __init__(self) -> None:
        self._tokens: dict[str, dict[str, Any]] = {}

    def create(
        self,
        issue_key: str,
        filename: str,
        mime_type: str,
        content: bytes,
        expires_at: datetime,
    ) -> str:
        self.sweep_expired()
        token = secrets.token_urlsafe(24)
        self._tokens[token] = {
            "issue_key": issue_key,
            "filename": filename,
            "mime_type": mime_type,
            "content": content,
            "expires_at": expires_at,
        }
        return token

    def get(self, token: str) -> dict[str, Any] | None:
        entry = self._tokens.get(token)
        if not entry:
            return None
        if _utcnow() > entry["expires_at"]:
            self._tokens.pop(token, None)
            return None
        return dict(entry)

    def clear(self) -> None:
        self._tokens.clear()

    def sweep_expired(self) -> int:
        now = _utcnow()
        expired = [k for k, v in self._tokens.items() if now > v["expires_at"]]
        for token in expired:
            self._tokens.pop(token, None)
        return len(expired)


class FilesystemDownloadTokenStore(DownloadTokenStore):
    """Download token store backed by a (optionally shared) directory.

    Point ``root_dir`` at a volume shared by every instance so any instance can
    serve a token minted elsewhere. Each token is a ``<token>.bin`` (content)
    plus ``<token>.json`` (metadata + expiry) pair.
    """

    def __init__(self, root_dir: str) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        issue_key: str,
        filename: str,
        mime_type: str,
        content: bytes,
        expires_at: datetime,
    ) -> str:
        token = secrets.token_urlsafe(24)
        self._atomic_write_bytes(self._root / f"{token}.bin", content)
        self._atomic_write_bytes(
            self._root / f"{token}.json",
            json.dumps(
                {
                    "issue_key": issue_key,
                    "filename": filename,
                    "mime_type": mime_type,
                    "expires_at": expires_at.isoformat(),
                }
            ).encode("utf-8"),
        )
        return token

    def get(self, token: str) -> dict[str, Any] | None:
        if not _TOKEN_RE.match(token):
            return None
        meta_path = self._root / f"{token}.json"
        bin_path = self._root / f"{token}.bin"
        if not meta_path.is_file() or not bin_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        expires_at = _parse_dt(meta.get("expires_at"))
        if expires_at is None or _utcnow() > expires_at:
            self._remove_token(token)
            return None
        try:
            content = bin_path.read_bytes()
        except OSError:
            return None
        return {
            "issue_key": meta.get("issue_key"),
            "filename": meta.get("filename"),
            "mime_type": meta.get("mime_type"),
            "content": content,
            "expires_at": expires_at,
        }

    def clear(self) -> None:
        for child in self._root.iterdir():
            if child.is_file() and child.suffix in (".bin", ".json"):
                child.unlink(missing_ok=True)

    def sweep_expired(self) -> int:
        removed = 0
        now = _utcnow()
        try:
            metas = list(self._root.glob("*.json"))
        except OSError:
            return 0
        for meta_path in metas:
            token = meta_path.stem
            if not _TOKEN_RE.match(token):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = None
            expires_at = _parse_dt(meta.get("expires_at")) if meta else None
            if expires_at is None or now > expires_at:
                self._remove_token(token)
                removed += 1
        # Reap temp files orphaned by interrupted atomic writes, guarded by age
        # so an in-flight write on another instance is never deleted mid-flight.
        cutoff = now - timedelta(hours=1)
        for tmp in self._root.glob("*.tmp"):
            try:
                mtime = datetime.fromtimestamp(tmp.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mtime < cutoff:
                tmp.unlink(missing_ok=True)
        return removed

    def _remove_token(self, token: str) -> None:
        for suffix in (".bin", ".json"):
            (self._root / f"{token}{suffix}").unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


def _build_download_token_store() -> DownloadTokenStore:
    """Build the configured download token store from environment settings.

    ``ATTACHMENT_DOWNLOAD_BACKEND`` selects the backend:
      - ``memory`` (default): process-local store (single instance).
      - ``filesystem``: shared dir from ``ATTACHMENT_DOWNLOAD_DIR`` (multi-instance).
      - ``package.module:ClassName``: a user-provided store (e.g. S3/DynamoDB).
    """
    raw = os.environ.get("ATTACHMENT_DOWNLOAD_BACKEND", "memory").strip()
    if ":" in raw:
        module_name, _, class_name = raw.partition(":")
        if not module_name or not class_name:
            raise ValueError(
                f"Invalid ATTACHMENT_DOWNLOAD_BACKEND path '{raw}'. "
                "Expected format 'package.module:ClassName'."
            )
        module = importlib.import_module(module_name)
        store = getattr(module, class_name)()
        if not isinstance(store, DownloadTokenStore):
            raise TypeError(
                f"{raw} must be a subclass of DownloadTokenStore, "
                f"got {type(store).__name__}."
            )
        return store

    name = raw.lower()
    if name in ("", "memory", "in-memory", "inmemory"):
        return MemoryDownloadTokenStore()
    if name in ("filesystem", "fs", "disk"):
        root_dir = os.environ.get("ATTACHMENT_DOWNLOAD_DIR")
        if not root_dir:
            raise ValueError(
                "ATTACHMENT_DOWNLOAD_BACKEND=filesystem requires "
                "ATTACHMENT_DOWNLOAD_DIR to point at a (shared) directory."
            )
        return FilesystemDownloadTokenStore(root_dir)
    raise ValueError(
        f"Unknown ATTACHMENT_DOWNLOAD_BACKEND '{raw}'. "
        "Expected 'memory', 'filesystem', or 'package.module:ClassName'."
    )


class AttachmentCache:
    """Thread-safe in-memory cache for storing attachment data temporarily."""

    def __init__(
        self,
        ttl_minutes: int = 10,
        max_size_mb: int = 100,
        token_store: DownloadTokenStore | None = None,
    ) -> None:
        """
        Initialize the attachment cache.

        Args:
            ttl_minutes: Time-to-live for cached items in minutes (default: 10)
            max_size_mb: Maximum total cache size in MB (default: 100)
            token_store: Backend for download tokens (default: in-memory)
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._ttl_minutes = ttl_minutes
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0
        self._remove_listeners: list[Callable[[str, str], None]] = []
        self._token_store = token_store or MemoryDownloadTokenStore()

    def add_remove_listener(self, listener: Callable[[str, str], None]) -> None:
        """Register a callback for when the last cached copy of a file is removed."""
        if listener not in self._remove_listeners:
            self._remove_listeners.append(listener)

    def _generate_key(self, issue_key: str, filename: str) -> str:
        """Generate a unique cache key for an attachment."""
        content = f"{issue_key}:{filename}:{_utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        now = _utcnow()
        expired_keys = [
            key for key, value in self._cache.items() if now > value["expires_at"]
        ]
        for key in expired_keys:
            self._remove(key)

    def _evict_lru(self) -> None:
        """Remove least recently used entries to free space."""
        if not self._cache:
            return

        # Sort by last accessed time
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1]["last_accessed"])

        # Remove oldest 20% of items
        to_remove = max(1, len(sorted_items) // 5)
        for key, _ in sorted_items[:to_remove]:
            self._remove(key)

    def _remove(self, key: str) -> None:
        """Remove an item from cache."""
        if key in self._cache:
            item = self._cache[key]
            issue_key = item["issue_key"]
            filename = item["filename"]
            self._current_size_bytes -= len(item["content"])
            del self._cache[key]
            logger.debug(f"Removed cached attachment: {key}")

            has_remaining_copy = any(
                cached_item["issue_key"] == issue_key
                and cached_item["filename"] == filename
                for cached_item in self._cache.values()
            )
            if not has_remaining_copy:
                for listener in self._remove_listeners:
                    try:
                        listener(issue_key, filename)
                    except Exception as exc:
                        logger.warning(
                            "Attachment cache remove listener failed for %s/%s: %s",
                            issue_key,
                            filename,
                            exc,
                        )

    def _get_latest_match(
        self, issue_key: str, filename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Return the newest cached entry for an issue/filename pair."""
        self._evict_expired()

        matches = [
            (key, item)
            for key, item in self._cache.items()
            if item["issue_key"] == issue_key and item["filename"] == filename
        ]

        if not matches:
            logger.debug(f"Cache miss for issue={issue_key}, filename={filename}")
            return None

        matches.sort(key=lambda x: x[1]["created_at"], reverse=True)
        return matches[0]

    def store(
        self, issue_key: str, filename: str, content: bytes, mime_type: str
    ) -> str:
        """
        Store attachment content in cache.

        Args:
            issue_key: The Jira issue key
            filename: The attachment filename
            content: The binary content of the attachment
            mime_type: The MIME type of the attachment

        Returns:
            The cache key/ID for retrieving the attachment
        """
        # Clean up expired entries first
        self._evict_expired()

        # Check if we need to make space
        content_size = len(content)
        while (
            self._current_size_bytes + content_size > self._max_size_bytes
            and self._cache
        ):
            self._evict_lru()

        # If still too large after eviction, reject
        if content_size > self._max_size_bytes:
            logger.error(
                f"Attachment {filename} is too large ({content_size} bytes) for cache"
            )
            raise ValueError(
                f"Attachment size ({content_size} bytes) exceeds cache limit"
            )

        # Generate unique key and store
        cache_key = self._generate_key(issue_key, filename)
        expires_at = _utcnow() + timedelta(minutes=self._ttl_minutes)

        self._cache[cache_key] = {
            "issue_key": issue_key,
            "filename": filename,
            "content": content,
            "mime_type": mime_type,
            "created_at": _utcnow(),
            "last_accessed": _utcnow(),
            "expires_at": expires_at,
            "size": content_size,
        }

        self._current_size_bytes += content_size
        logger.info(
            f"Cached attachment {filename} from {issue_key} (key: {cache_key}, "
            f"size: {content_size} bytes, cache usage: {self._current_size_bytes}/{self._max_size_bytes})"
        )

        return cache_key

    def get_by_issue_and_filename(
        self, issue_key: str, filename: str
    ) -> dict[str, Any] | None:
        """
        Retrieve the most recently cached attachment by issue key and filename.

        Args:
            issue_key: The Jira issue key
            filename: The attachment filename

        Returns:
            Dictionary with 'content', 'mime_type', 'filename', 'issue_key' or None if not found
        """
        match = self._get_latest_match(issue_key, filename)

        if not match:
            return None

        key, item = match
        item["last_accessed"] = _utcnow()

        logger.debug(
            f"Cache hit for issue={issue_key}, filename={filename} (key: {key})"
        )
        return {
            "content": item["content"],
            "mime_type": item["mime_type"],
            "filename": item["filename"],
            "issue_key": item["issue_key"],
        }

    def create_download_token(
        self, issue_key: str, filename: str, ttl_minutes: int = 5
    ) -> dict[str, Any]:
        """Create a short-lived token for downloading a cached attachment over HTTP."""
        match = self._get_latest_match(issue_key, filename)
        if not match:
            raise ValueError(
                f"Attachment '{filename}' for issue '{issue_key}' is not cached. "
                "Run jira_download_attachments with return_content=true first."
            )

        _, item = match
        requested_expiry = _utcnow() + timedelta(minutes=ttl_minutes)
        expires_at = min(requested_expiry, item["expires_at"])
        token = self._token_store.create(
            issue_key=issue_key,
            filename=filename,
            mime_type=item["mime_type"],
            content=item["content"],
            expires_at=expires_at,
        )

        return {
            "token": token,
            "expires_at": expires_at,
            "mime_type": item["mime_type"],
            "filename": item["filename"],
            "issue_key": item["issue_key"],
        }

    def get_by_download_token(self, token: str) -> dict[str, Any] | None:
        """Resolve a download token to its self-contained attachment content."""
        entry = self._token_store.get(token)
        if not entry:
            logger.debug(f"Download token miss: {token}")
            return None

        return {
            "content": entry["content"],
            "mime_type": entry["mime_type"],
            "filename": entry["filename"],
            "issue_key": entry["issue_key"],
            "expires_at": entry["expires_at"],
        }

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """
        Retrieve attachment content from cache.

        Args:
            cache_key: The cache key returned from store()

        Returns:
            Dictionary with 'content', 'mime_type', 'filename', 'issue_key' or None if not found
        """
        self._evict_expired()

        if cache_key not in self._cache:
            logger.debug(f"Cache miss for key: {cache_key}")
            return None

        item = self._cache[cache_key]
        item["last_accessed"] = _utcnow()

        logger.debug(f"Cache hit for key: {cache_key} (file: {item['filename']})")
        return {
            "content": item["content"],
            "mime_type": item["mime_type"],
            "filename": item["filename"],
            "issue_key": item["issue_key"],
        }

    def clear(self) -> None:
        """Clear all cached attachments."""
        count = len(self._cache)
        for key in list(self._cache):
            self._remove(key)
        self._token_store.clear()
        logger.info(f"Cleared attachment cache ({count} items)")

    def sweep_expired(self) -> int:
        """Proactively evict expired cache entries and download tokens.

        The per-read checks only reclaim entries that are accessed again; this
        also reclaims download tokens (and their shared-disk files) that are
        minted but never fetched.
        """
        before = len(self._cache)
        self._evict_expired()
        removed = before - len(self._cache)
        removed += self._token_store.sweep_expired()
        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        self._evict_expired()
        return {
            "item_count": len(self._cache),
            "total_size_bytes": self._current_size_bytes,
            "max_size_bytes": self._max_size_bytes,
            "utilization_percent": round(
                (self._current_size_bytes / self._max_size_bytes) * 100, 2
            )
            if self._max_size_bytes > 0
            else 0,
        }


# Global cache instance
_attachment_cache = AttachmentCache(token_store=_build_download_token_store())


def get_attachment_cache() -> AttachmentCache:
    """Get the global attachment cache instance."""
    return _attachment_cache
