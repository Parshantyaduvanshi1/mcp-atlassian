"""In-memory staging store for client-uploaded files pending Jira attachment.

Files are uploaded by the MCP client to the /upload HTTP endpoint, staged here,
then consumed by the jira_upload_attachment MCP tool to push them to Jira.

URI scheme: upload://sessions/<session_id>/<file_id>
"""

import importlib
import json
import logging
import os
import re
import secrets
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("mcp-jira")

# Server-issued session/file identifiers are URL-safe tokens. Validate them
# before using them as path components to prevent traversal on disk backends.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _safe_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid value for %s=%r, falling back to default %d", name, raw, default
        )
        return default


_TTL_MINUTES = _safe_int_env("UPLOAD_STAGING_TTL_MINUTES", 30)
_MAX_SIZE_MB = _safe_int_env("UPLOAD_STAGING_MAX_SIZE_MB", 200)
_URI_PREFIX = "upload://sessions/"


class UploadStagingBackend(ABC):
    """Abstract base for pluggable upload staging backends.

    A backend mints opaque session tokens, stores file bytes against a
    session, and retrieves them later for the ``jira_upload_attachment`` tool.
    Implementations that persist to shared storage (e.g. a shared filesystem or
    object store) allow the MCP server to run statelessly behind a round-robin
    load balancer.
    """

    @property
    @abstractmethod
    def max_file_bytes(self) -> int:
        """Maximum size, in bytes, allowed for a single staged file."""

    @abstractmethod
    def create_session(self) -> str:
        """Generate and persist a new opaque upload session token."""

    @abstractmethod
    def is_valid_session(self, session_id: str) -> bool:
        """Return True when the session was issued here and is unexpired."""

    @abstractmethod
    def store(
        self, session_id: str, filename: str, content: bytes, mime_type: str
    ) -> str:
        """Stage a file and return its file_id."""

    @abstractmethod
    def get(self, session_id: str, file_id: str) -> dict[str, Any] | None:
        """Return a staged entry, or None if not found / expired."""

    @abstractmethod
    def remove(self, session_id: str, file_id: str) -> None:
        """Remove a staged file (call after a successful Jira upload)."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all staged files and issued sessions."""

    @abstractmethod
    def sweep_expired(self) -> int:
        """Delete expired sessions/files; return the number of files removed."""

    @staticmethod
    def make_uri(session_id: str, file_id: str) -> str:
        """Build the canonical upload:// URI for a staged file."""
        return f"{_URI_PREFIX}{session_id}/{file_id}"

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str] | None:
        """Parse upload://sessions/<session_id>/<file_id> → (session_id, file_id).

        Returns None if the URI does not match the expected format.
        """
        if not uri.startswith(_URI_PREFIX):
            return None
        rest = uri[len(_URI_PREFIX) :]
        parts = rest.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return parts[0], parts[1]


class UploadStagingStore(UploadStagingBackend):
    """In-memory staging store for files uploaded by MCP clients."""

    def __init__(self, ttl_minutes: int = 30, max_size_mb: int = 200) -> None:
        # {session_id: {file_id: entry_dict}}
        self._store: dict[str, dict[str, dict[str, Any]]] = {}
        self._sessions: dict[str, datetime] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0

    @property
    def max_file_bytes(self) -> int:
        return self._max_size_bytes

    def create_session(self) -> str:
        """Generate a new opaque upload session token."""
        session_id = secrets.token_urlsafe(24)
        self._sessions[session_id] = _utcnow() + self._ttl
        return session_id

    def is_valid_session(self, session_id: str) -> bool:
        """Return True when the session was issued by the server and is unexpired."""
        self._evict_expired()
        expires_at = self._sessions.get(session_id)
        return expires_at is not None and _utcnow() <= expires_at

    def store(
        self, session_id: str, filename: str, content: bytes, mime_type: str
    ) -> str:
        """Stage a file and return its file_id.

        Args:
            session_id: The upload session token (from construct_upload_endpoint).
            filename: Original filename (already sanitised by the upload endpoint).
            content: Raw file bytes.
            mime_type: MIME type of the file.

        Returns:
            file_id component of the resulting upload:// URI.

        Raises:
            PermissionError: If the session was not issued by the server or expired.
            ValueError: If the file is too large for the staging store.
        """
        self._evict_expired()

        if not self.is_valid_session(session_id):
            raise PermissionError(
                "Upload session is invalid or expired. "
                "Call construct_upload_endpoint to create a new session."
            )

        content_size = len(content)
        if content_size > self._max_size_bytes:
            raise ValueError(
                f"File size ({content_size} bytes) exceeds staging limit "
                f"({self._max_size_bytes} bytes)"
            )

        # Evict LRU entries if needed to make room
        while (
            self._current_size_bytes + content_size > self._max_size_bytes
            and self._store
        ):
            self._evict_oldest()

        file_id = secrets.token_urlsafe(12)
        entry: dict[str, Any] = {
            "filename": filename,
            "content": content,
            "mime_type": mime_type,
            "created_at": _utcnow(),
            "expires_at": _utcnow() + self._ttl,
        }
        self._store.setdefault(session_id, {})[file_id] = entry
        self._current_size_bytes += content_size
        logger.debug(
            "Staged upload '%s' → session=%s file_id=%s (%d bytes)",
            filename,
            session_id,
            file_id,
            content_size,
        )
        return file_id

    def get(self, session_id: str, file_id: str) -> dict[str, Any] | None:
        """Return a staged entry, or None if not found / expired."""
        self._evict_expired()
        entry = self._store.get(session_id, {}).get(file_id)
        if entry and _utcnow() <= entry["expires_at"]:
            return entry
        return None

    def remove(self, session_id: str, file_id: str) -> None:
        """Remove a staged file (call after successful Jira upload)."""
        session = self._store.get(session_id, {})
        if file_id in session:
            self._current_size_bytes -= len(session[file_id]["content"])
            del session[file_id]
            if not session:
                del self._store[session_id]
                logger.debug("Removed empty upload session: %s", session_id)

    def clear(self) -> None:
        """Clear all staged files and issued sessions."""
        self._store.clear()
        self._sessions.clear()
        self._current_size_bytes = 0

    def sweep_expired(self) -> int:
        """Delete expired sessions/files; return the number of files removed."""
        before = sum(len(files) for files in self._store.values())
        self._evict_expired()
        after = sum(len(files) for files in self._store.values())
        return before - after

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        now = _utcnow()
        expired_sessions = [
            session_id
            for session_id, expires_at in self._sessions.items()
            if now > expires_at
        ]
        for session_id in expired_sessions:
            session = self._store.pop(session_id, {})
            for entry in session.values():
                self._current_size_bytes -= len(entry["content"])
            self._sessions.pop(session_id, None)

        for session_id in list(self._store):
            for file_id in list(self._store[session_id]):
                if now > self._store[session_id][file_id]["expires_at"]:
                    self._current_size_bytes -= len(
                        self._store[session_id][file_id]["content"]
                    )
                    del self._store[session_id][file_id]
            if not self._store.get(session_id):
                self._store.pop(session_id, None)

    def _evict_oldest(self) -> None:
        """Remove the globally oldest staged file to free space."""
        oldest_key: tuple[str, str] | None = None
        oldest_time: datetime | None = None
        for session_id, files in self._store.items():
            for file_id, entry in files.items():
                if oldest_time is None or entry["created_at"] < oldest_time:
                    oldest_time = entry["created_at"]
                    oldest_key = (session_id, file_id)
        if oldest_key:
            self.remove(*oldest_key)


class FilesystemUploadStagingBackend(UploadStagingBackend):
    """Staging backend that persists files to a (optionally shared) directory.

    Point ``root_dir`` at a volume shared by every server instance (e.g. NFS or
    an EFS mount) so any instance behind a round-robin load balancer can serve a
    staged upload. Each session is a subdirectory holding a ``session.json``
    marker plus ``<file_id>.bin`` / ``<file_id>.json`` pairs.
    """

    def __init__(
        self, root_dir: str, ttl_minutes: int = 30, max_size_mb: int = 200
    ) -> None:
        self._root = Path(root_dir)
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def max_file_bytes(self) -> int:
        return self._max_size_bytes

    def _session_dir(self, session_id: str) -> Path | None:
        if not _TOKEN_RE.match(session_id):
            return None
        return self._root / session_id

    def create_session(self) -> str:
        session_id = secrets.token_urlsafe(24)
        session_dir = self._root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        expires_at = _utcnow() + self._ttl
        self._write_json(
            session_dir / "session.json", {"expires_at": expires_at.isoformat()}
        )
        return session_id

    def is_valid_session(self, session_id: str) -> bool:
        session_dir = self._session_dir(session_id)
        if session_dir is None:
            return False
        marker = session_dir / "session.json"
        if not marker.is_file():
            return False
        data = self._read_json(marker)
        if data is None:
            return False
        expires_at = self._parse_dt(data.get("expires_at"))
        if expires_at is None or _utcnow() > expires_at:
            self._remove_session_dir(session_dir)
            return False
        return True

    def store(
        self, session_id: str, filename: str, content: bytes, mime_type: str
    ) -> str:
        if not self.is_valid_session(session_id):
            raise PermissionError(
                "Upload session is invalid or expired. "
                "Call construct_upload_endpoint to create a new session."
            )

        content_size = len(content)
        if content_size > self._max_size_bytes:
            raise ValueError(
                f"File size ({content_size} bytes) exceeds staging limit "
                f"({self._max_size_bytes} bytes)"
            )

        session_dir = self._session_dir(session_id)
        if session_dir is None:  # pragma: no cover - guarded by is_valid_session
            raise PermissionError("Invalid upload session identifier.")
        file_id = secrets.token_urlsafe(12)
        created_at = _utcnow()
        expires_at = created_at + self._ttl
        self._atomic_write_bytes(session_dir / f"{file_id}.bin", content)
        self._write_json(
            session_dir / f"{file_id}.json",
            {
                "filename": filename,
                "mime_type": mime_type,
                "created_at": created_at.isoformat(),
                "expires_at": expires_at.isoformat(),
            },
        )
        logger.debug(
            "Staged upload '%s' \u2192 session=%s file_id=%s (%d bytes) [filesystem]",
            filename,
            session_id,
            file_id,
            content_size,
        )
        return file_id

    def get(self, session_id: str, file_id: str) -> dict[str, Any] | None:
        session_dir = self._session_dir(session_id)
        if session_dir is None or not _TOKEN_RE.match(file_id):
            return None
        meta_path = session_dir / f"{file_id}.json"
        bin_path = session_dir / f"{file_id}.bin"
        if not meta_path.is_file() or not bin_path.is_file():
            return None
        meta = self._read_json(meta_path)
        if meta is None:
            return None
        expires_at = self._parse_dt(meta.get("expires_at"))
        if expires_at is None or _utcnow() > expires_at:
            self.remove(session_id, file_id)
            return None
        try:
            content = bin_path.read_bytes()
        except OSError:
            return None
        return {
            "filename": meta.get("filename"),
            "content": content,
            "mime_type": meta.get("mime_type"),
            "created_at": self._parse_dt(meta.get("created_at")),
            "expires_at": expires_at,
        }

    def remove(self, session_id: str, file_id: str) -> None:
        session_dir = self._session_dir(session_id)
        if session_dir is None or not _TOKEN_RE.match(file_id):
            return
        for suffix in (".bin", ".json"):
            path = session_dir / f"{file_id}{suffix}"
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        for child in self._root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    def sweep_expired(self) -> int:
        """Delete expired sessions/files; return the number of files removed."""
        if not self._root.exists():
            return 0
        removed = 0
        now = _utcnow()
        try:
            children = list(self._root.iterdir())
        except OSError:
            return 0
        for session_dir in children:
            if not session_dir.is_dir():
                continue
            marker = session_dir / "session.json"
            data = self._read_json(marker) if marker.is_file() else None
            session_expires = self._parse_dt(data.get("expires_at")) if data else None
            # Whole session expired (or marker missing/corrupt): drop the dir.
            if session_expires is None or now > session_expires:
                removed += sum(1 for _ in session_dir.glob("*.bin"))
                self._remove_session_dir(session_dir)
                continue
            # Session still valid: drop any individually-expired files.
            for meta_path in session_dir.glob("*.json"):
                if meta_path.name == "session.json":
                    continue
                meta = self._read_json(meta_path)
                file_expires = self._parse_dt(meta.get("expires_at")) if meta else None
                if file_expires is None or now > file_expires:
                    self.remove(session_dir.name, meta_path.stem)
                    removed += 1
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dt(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        FilesystemUploadStagingBackend._atomic_write_bytes(
            path, json.dumps(data).encode("utf-8")
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    @staticmethod
    def _remove_session_dir(session_dir: Path) -> None:
        shutil.rmtree(session_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_upload_staging: UploadStagingBackend | None = None


def _load_backend_from_path(path: str) -> UploadStagingBackend:
    """Instantiate a user-provided backend given a ``pkg.module:ClassName`` path."""
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ValueError(
            f"Invalid UPLOAD_STAGING_BACKEND path '{path}'. "
            "Expected format 'package.module:ClassName'."
        )
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, class_name)
    instance = backend_cls(ttl_minutes=_TTL_MINUTES, max_size_mb=_MAX_SIZE_MB)
    if not isinstance(instance, UploadStagingBackend):
        raise TypeError(
            f"{path} must be a subclass of UploadStagingBackend, "
            f"got {type(instance).__name__}."
        )
    return instance


def _build_upload_staging() -> UploadStagingBackend:
    """Build the configured upload staging backend from environment settings.

    ``UPLOAD_STAGING_BACKEND`` selects the backend:
      - ``memory`` (default): process-local in-memory store (single instance).
      - ``filesystem``: shared directory from ``UPLOAD_STAGING_DIR`` (stateless).
      - ``package.module:ClassName``: a user-provided backend (e.g. S3), enabling
        stateless deployments without bundling heavy dependencies in core.
    """
    raw = os.environ.get("UPLOAD_STAGING_BACKEND", "memory").strip()
    if ":" in raw:
        return _load_backend_from_path(raw)

    name = raw.lower()
    if name in ("", "memory", "in-memory", "inmemory"):
        return UploadStagingStore(ttl_minutes=_TTL_MINUTES, max_size_mb=_MAX_SIZE_MB)
    if name in ("filesystem", "fs", "disk"):
        root_dir = os.environ.get("UPLOAD_STAGING_DIR")
        if not root_dir:
            raise ValueError(
                "UPLOAD_STAGING_BACKEND=filesystem requires UPLOAD_STAGING_DIR "
                "to point at a (shared) directory."
            )
        return FilesystemUploadStagingBackend(
            root_dir=root_dir,
            ttl_minutes=_TTL_MINUTES,
            max_size_mb=_MAX_SIZE_MB,
        )
    raise ValueError(
        f"Unknown UPLOAD_STAGING_BACKEND '{raw}'. "
        "Expected 'memory', 'filesystem', or 'package.module:ClassName'."
    )


def get_upload_staging() -> UploadStagingBackend:
    """Return the process-wide singleton upload staging backend."""
    global _upload_staging
    if _upload_staging is None:
        _upload_staging = _build_upload_staging()
    return _upload_staging
