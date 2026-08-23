"""Tests for the Jira attachments module."""

import os
import tempfile
from datetime import timedelta
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mcp_atlassian.jira import JiraFetcher
from mcp_atlassian.jira.attachment_cache import (
    AttachmentCache,
    DownloadTokenStore,
    FilesystemDownloadTokenStore,
    MemoryDownloadTokenStore,
    _build_download_token_store,
)
from mcp_atlassian.jira.attachment_cache import _utcnow as _cache_utcnow
from mcp_atlassian.jira.attachments import AttachmentsMixin
from mcp_atlassian.jira.upload_staging import (
    FilesystemUploadStagingBackend,
    UploadStagingBackend,
    UploadStagingStore,
)

# Test scenarios for AttachmentsMixin
#
# 1. Single Attachment Download (download_attachment method):
#    - Success case: Downloads attachment correctly with proper HTTP response
#    - Path handling: Converts relative path to absolute path
#    - Error cases:
#      - No URL provided
#      - HTTP error during download
#      - File write error
#      - File not created after write operation
#
# 2. Issue Attachments Download (download_issue_attachments method):
#    - Success case: Downloads all attachments for an issue
#    - Path handling: Converts relative target directory to absolute path
#    - Edge cases:
#      - Issue has no attachments
#      - Issue not found
#      - Issue has no fields
#      - Some attachments fail to download
#      - Attachment has missing URL
#
# 3. Single Attachment Upload (upload_attachment method):
#    - Success case: Uploads file correctly
#    - Path handling: Converts relative file path to absolute path
#    - Error cases:
#      - No issue key provided
#      - No file path provided
#      - File not found
#      - API error during upload
#      - No response from API
#
# 4. Multiple Attachments Upload (upload_attachments method):
#    - Success case: Uploads multiple files correctly
#    - Partial success: Some files upload successfully, others fail
#    - Error cases:
#      - Empty list of file paths
#      - No issue key provided


class TestAttachmentsMixin:
    """Tests for the AttachmentsMixin class."""

    @pytest.fixture
    def attachments_mixin(self, jira_fetcher: JiraFetcher) -> AttachmentsMixin:
        """Set up test fixtures before each test method."""
        # Create a mock Jira client
        attachments_mixin = jira_fetcher
        attachments_mixin.jira = MagicMock()
        attachments_mixin.jira._session = MagicMock()
        return attachments_mixin

    def test_download_attachment_success(self, attachments_mixin: AttachmentsMixin):
        """Test successful attachment download."""
        test_file = os.path.join(tempfile.gettempdir(), "test_file.txt")

        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # Mock file operations
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("os.makedirs") as mock_makedirs,
        ):
            mock_exists.return_value = True
            mock_getsize.return_value = 12  # Length of "test content"

            # Call the method
            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", test_file
            )

            # Assertions
            assert result is True
            attachments_mixin.jira._session.get.assert_called_once_with(
                "https://test.url/attachment", stream=True
            )
            mock_file.assert_called_once_with(test_file, "wb")
            mock_file().write.assert_called_once_with(b"test content")
            mock_makedirs.assert_called_once()

    def test_download_attachment_relative_path(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment download with a relative path."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # Mock file operations and os.path.abspath
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("os.makedirs") as mock_makedirs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.isabs") as mock_isabs,
        ):
            mock_exists.return_value = True
            mock_getsize.return_value = 12
            mock_isabs.return_value = False
            mock_abspath.return_value = "/absolute/path/test_file.txt"

            # Call the method with a relative path
            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", "test_file.txt"
            )

            # Assertions
            assert result is True
            mock_isabs.assert_called_once_with("test_file.txt")
            mock_abspath.assert_called_once_with("test_file.txt")
            mock_file.assert_called_once_with("/absolute/path/test_file.txt", "wb")

    def test_download_attachment_no_url(self, attachments_mixin: AttachmentsMixin):
        """Test attachment download with no URL."""
        result = attachments_mixin.download_attachment("", "/tmp/test_file.txt")
        assert result is False

    def test_download_attachment_http_error(self, attachments_mixin: AttachmentsMixin):
        """Test attachment download with an HTTP error."""
        # Mock the response to raise an HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        attachments_mixin.jira._session.get.return_value = mock_response

        result = attachments_mixin.download_attachment(
            "https://test.url/attachment", "/tmp/test_file.txt"
        )
        assert result is False

    def test_download_attachment_file_write_error(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment download with a file write error."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # Mock file operations to raise an exception during write
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.makedirs") as mock_makedirs,
        ):
            mock_file().write.side_effect = OSError("Write error")

            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", "/tmp/test_file.txt"
            )
            assert result is False

    def test_download_attachment_file_not_created(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment download when file is not created."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_response.raise_for_status = MagicMock()
        attachments_mixin.jira._session.get.return_value = mock_response

        # Mock file operations
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.path.exists") as mock_exists,
            patch("os.makedirs") as mock_makedirs,
        ):
            mock_exists.return_value = False  # File doesn't exist after write

            result = attachments_mixin.download_attachment(
                "https://test.url/attachment", "/tmp/test_file.txt"
            )
            assert result is False

    def test_download_issue_attachments_success(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test successful download of all issue attachments."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    },
                    {
                        "filename": "test2.txt",
                        "content": "https://test.url/attachment2",
                        "size": 200,
                    },
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock JiraAttachment.from_api_response
        mock_attachment1 = MagicMock()
        mock_attachment1.filename = "test1.txt"
        mock_attachment1.url = "https://test.url/attachment1"
        mock_attachment1.size = 100

        mock_attachment2 = MagicMock()
        mock_attachment2.filename = "test2.txt"
        mock_attachment2.url = "https://test.url/attachment2"
        mock_attachment2.size = 200

        # Mock the download_attachment method
        with (
            patch.object(
                attachments_mixin, "download_attachment", return_value=True
            ) as mock_download,
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                side_effect=[mock_attachment1, mock_attachment2],
            ),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "/tmp/attachments"
            )

            # Assertions
            assert result["success"] is True
            assert len(result["downloaded"]) == 2
            assert len(result["failed"]) == 0
            assert result["total"] == 2
            assert result["issue_key"] == "TEST-123"
            assert mock_download.call_count == 2
            mock_mkdir.assert_called_once()

    def test_download_issue_attachments_relative_path(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download issue attachments with a relative path."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock attachment
        mock_attachment = MagicMock()
        mock_attachment.filename = "test1.txt"
        mock_attachment.url = "https://test.url/attachment1"
        mock_attachment.size = 100

        # Mock path operations
        with (
            patch.object(
                attachments_mixin, "download_attachment", return_value=True
            ) as mock_download,
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
        ):
            mock_isabs.return_value = False
            mock_abspath.return_value = "/absolute/path/attachments"

            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "attachments"
            )

            # Assertions
            assert result["success"] is True
            mock_isabs.assert_called_once_with("attachments")
            mock_abspath.assert_called_once_with("attachments")

    def test_download_issue_attachments_no_attachments(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download when issue has no attachments."""
        # Mock the issue data with no attachments
        mock_issue = {"fields": {"attachment": []}}
        attachments_mixin.jira.issue.return_value = mock_issue

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "/tmp/attachments"
            )

            # Assertions
            assert result["success"] is True
            assert "No attachments found" in result["message"]
            assert len(result["downloaded"]) == 0
            assert len(result["failed"]) == 0
            mock_mkdir.assert_called_once()

    def test_download_issue_attachments_issue_not_found(
        self, attachments_mixin: AttachmentsMixin, tmp_path
    ):
        """Test download when issue cannot be retrieved."""
        attachments_mixin.jira.issue.return_value = None

        with pytest.raises(
            TypeError,
            match="Unexpected return value type from `jira.issue`: <class 'NoneType'>",
        ):
            attachments_mixin.download_issue_attachments(
                "TEST-123", str(tmp_path / "attachments")
            )

    def test_download_issue_attachments_no_fields(
        self, attachments_mixin: AttachmentsMixin, tmp_path
    ):
        """Test download when issue has no fields."""
        # Mock the issue data with no fields
        mock_issue = {}  # Missing 'fields' key
        attachments_mixin.jira.issue.return_value = mock_issue

        result = attachments_mixin.download_issue_attachments(
            "TEST-123", str(tmp_path / "attachments")
        )

        # Assertions
        assert result["success"] is False
        assert "Could not retrieve issue" in result["error"]

    def test_download_issue_attachments_some_failures(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download when some attachments fail to download."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    },
                    {
                        "filename": "test2.txt",
                        "content": "https://test.url/attachment2",
                        "size": 200,
                    },
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock attachments
        mock_attachment1 = MagicMock()
        mock_attachment1.filename = "test1.txt"
        mock_attachment1.url = "https://test.url/attachment1"
        mock_attachment1.size = 100

        mock_attachment2 = MagicMock()
        mock_attachment2.filename = "test2.txt"
        mock_attachment2.url = "https://test.url/attachment2"
        mock_attachment2.size = 200

        # Mock the download_attachment method to succeed for first attachment and fail for second
        with (
            patch.object(
                attachments_mixin, "download_attachment", side_effect=[True, False]
            ) as mock_download,
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                side_effect=[mock_attachment1, mock_attachment2],
            ),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "/tmp/attachments"
            )

            # Assertions
            assert result["success"] is True
            assert len(result["downloaded"]) == 1
            assert len(result["failed"]) == 1
            assert result["downloaded"][0]["filename"] == "test1.txt"
            assert result["failed"][0]["filename"] == "test2.txt"
            assert mock_download.call_count == 2

    def test_download_issue_attachments_missing_url(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test download when an attachment has no URL."""
        # Mock the issue data
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        # Mock attachment with no URL
        mock_attachment = MagicMock()
        mock_attachment.filename = "test1.txt"
        mock_attachment.url = None  # No URL
        mock_attachment.size = 100

        # Mock path operations
        with (
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", "/tmp/attachments"
            )

            # Assertions
            assert result["success"] is True
            assert len(result["downloaded"]) == 0
            assert len(result["failed"]) == 1
            assert result["failed"][0]["filename"] == "test1.txt"
            assert "No URL available" in result["failed"][0]["error"]

    def test_download_issue_attachments_cache_failure_reports_error(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test cache failures return an error instead of embedding base64 content."""
        mock_issue = {
            "fields": {
                "attachment": [
                    {
                        "filename": "test1.txt",
                        "content": "https://test.url/attachment1",
                        "size": 100,
                    }
                ]
            }
        }
        attachments_mixin.jira.issue.return_value = mock_issue

        mock_attachment = MagicMock()
        mock_attachment.filename = "test1.txt"
        mock_attachment.url = "https://test.url/attachment1"
        mock_attachment.size = 100
        mock_attachment.content_type = "text/plain"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        attachments_mixin.jira._session.get.return_value = mock_response

        cache = MagicMock()
        cache.store.side_effect = ValueError("cache full")

        with (
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
            patch(
                "mcp_atlassian.jira.attachments.get_attachment_cache",
                return_value=cache,
            ),
        ):
            result = attachments_mixin.download_issue_attachments(
                "TEST-123", return_content=True
            )

        assert result["downloaded"] == []
        assert len(result["failed"]) == 1
        assert "Failed to cache attachment content" in result["failed"][0]["error"]
        assert "content" not in result["failed"][0]

    def test_fetch_and_cache_attachment_success(self, attachments_mixin):
        """fetch_and_cache_attachment downloads a single file and stores it."""
        attachments_mixin.jira.issue.return_value = {
            "fields": {"attachment": [{"filename": "report.pdf"}]}
        }

        mock_attachment = MagicMock()
        mock_attachment.filename = "report.pdf"
        mock_attachment.url = "https://test.url/report.pdf"
        mock_attachment.content_type = "application/pdf"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content.return_value = [b"pdf-", b"bytes"]
        attachments_mixin.jira._session.get.return_value = mock_response

        cache = MagicMock()
        with (
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
            patch(
                "mcp_atlassian.jira.attachments.get_attachment_cache",
                return_value=cache,
            ),
        ):
            result = attachments_mixin.fetch_and_cache_attachment(
                "TEST-123", "report.pdf"
            )

        assert result is True
        cache.store.assert_called_once_with(
            issue_key="TEST-123",
            filename="report.pdf",
            content=b"pdf-bytes",
            mime_type="application/pdf",
        )

    def test_fetch_and_cache_attachment_not_found(self, attachments_mixin):
        """fetch_and_cache_attachment returns False when the filename is absent."""
        attachments_mixin.jira.issue.return_value = {
            "fields": {"attachment": [{"filename": "other.txt"}]}
        }

        mock_attachment = MagicMock()
        mock_attachment.filename = "other.txt"
        mock_attachment.url = "https://test.url/other.txt"

        cache = MagicMock()
        with (
            patch(
                "mcp_atlassian.models.jira.JiraAttachment.from_api_response",
                return_value=mock_attachment,
            ),
            patch(
                "mcp_atlassian.jira.attachments.get_attachment_cache",
                return_value=cache,
            ),
        ):
            result = attachments_mixin.fetch_and_cache_attachment(
                "TEST-123", "missing.pdf"
            )

        assert result is False
        cache.store.assert_not_called()
        attachments_mixin.jira._session.get.assert_not_called()


class TestUploadStagingStore:
    """Tests for the upload staging session validation flow."""

    @pytest.fixture
    def attachments_mixin(self, jira_fetcher: JiraFetcher) -> AttachmentsMixin:
        """Mirror the shared AttachmentsMixin fixture for tests in this class."""
        attachments_mixin = jira_fetcher
        attachments_mixin.jira = MagicMock()
        attachments_mixin.jira._session = MagicMock()
        return attachments_mixin

    def test_store_requires_issued_session(self):
        """Test staging rejects arbitrary caller-provided session ids."""
        store = UploadStagingStore(ttl_minutes=30, max_size_mb=1)

        with pytest.raises(PermissionError, match="invalid or expired"):
            store.store("unknown-session", "test.txt", b"data", "text/plain")

    def test_store_accepts_server_issued_session(self):
        """Test staging accepts sessions created by create_session."""
        store = UploadStagingStore(ttl_minutes=30, max_size_mb=1)
        session_id = store.create_session()

        file_id = store.store(session_id, "test.txt", b"data", "text/plain")

        assert store.get(session_id, file_id) is not None


class TestAttachmentCacheDownloadTokens:
    """Tests for short-lived attachment download tokens."""

    @pytest.fixture
    def attachments_mixin(self, jira_fetcher: JiraFetcher) -> AttachmentsMixin:
        """Mirror the shared AttachmentsMixin fixture for tests in this class."""
        attachments_mixin = jira_fetcher
        attachments_mixin.jira = MagicMock()
        attachments_mixin.jira._session = MagicMock()
        return attachments_mixin

    def test_create_download_token_requires_cached_attachment(self):
        """Test a download URL cannot be created for an uncached attachment."""
        cache = AttachmentCache(ttl_minutes=10, max_size_mb=1)

        with pytest.raises(ValueError, match="is not cached"):
            cache.create_download_token("PROJ-1", "missing.txt")

    def test_download_token_resolves_cached_attachment(self):
        """Test a valid download token resolves to the cached attachment content."""
        cache = AttachmentCache(ttl_minutes=10, max_size_mb=1)
        cache.store(
            issue_key="PROJ-1",
            filename="report.pdf",
            content=b"pdf-bytes",
            mime_type="application/pdf",
        )

        token_info = cache.create_download_token("PROJ-1", "report.pdf", ttl_minutes=5)
        attachment = cache.get_by_download_token(token_info["token"])

        assert attachment is not None
        assert attachment["content"] == b"pdf-bytes"
        assert attachment["mime_type"] == "application/pdf"

    def test_clear_revokes_download_tokens(self):
        """Test clearing the cache invalidates outstanding download tokens."""
        cache = AttachmentCache(ttl_minutes=10, max_size_mb=1)
        cache.store(
            issue_key="PROJ-1",
            filename="report.pdf",
            content=b"pdf-bytes",
            mime_type="application/pdf",
        )
        token_info = cache.create_download_token("PROJ-1", "report.pdf", ttl_minutes=5)

        cache.clear()

        assert cache.get_by_download_token(token_info["token"]) is None

    # Tests for upload_attachment method

    def test_upload_attachment_success(self, attachments_mixin: AttachmentsMixin):
        """Test successful attachment upload."""
        # Mock the Jira API response
        mock_attachment_response = {
            "id": "12345",
            "filename": "test_file.txt",
            "size": 100,
        }
        attachments_mixin.jira.add_attachment.return_value = mock_attachment_response

        # Mock file operations
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.basename") as mock_basename,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = True
            mock_getsize.return_value = 100
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"
            mock_basename.return_value = "test_file.txt"

            # Call the method
            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is True
            assert result["issue_key"] == "TEST-123"
            assert result["filename"] == "test_file.txt"
            assert result["size"] == 100
            assert result["id"] == "12345"
            attachments_mixin.jira.add_attachment.assert_called_once_with(
                issue_key="TEST-123", filename="/absolute/path/test_file.txt"
            )

    def test_upload_attachment_relative_path(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload with a relative path."""
        # Mock the Jira API response
        mock_attachment_response = {
            "id": "12345",
            "filename": "test_file.txt",
            "size": 100,
        }
        attachments_mixin.jira.add_attachment.return_value = mock_attachment_response

        # Mock file operations
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.basename") as mock_basename,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = True
            mock_getsize.return_value = 100
            mock_isabs.return_value = False
            mock_abspath.return_value = "/absolute/path/test_file.txt"
            mock_basename.return_value = "test_file.txt"

            # Call the method with a relative path
            result = attachments_mixin.upload_attachment("TEST-123", "test_file.txt")

            # Assertions
            assert result["success"] is True
            mock_isabs.assert_called_once_with("test_file.txt")
            mock_abspath.assert_called_once_with("test_file.txt")
            attachments_mixin.jira.add_attachment.assert_called_once_with(
                issue_key="TEST-123", filename="/absolute/path/test_file.txt"
            )

    def test_upload_attachment_no_issue_key(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload with no issue key."""
        result = attachments_mixin.upload_attachment("", "/path/to/file.txt")

        # Assertions
        assert result["success"] is False
        assert "No issue key provided" in result["error"]
        attachments_mixin.jira.add_attachment.assert_not_called()

    def test_upload_attachment_no_file_path(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload with no file path."""
        result = attachments_mixin.upload_attachment("TEST-123", "")

        # Assertions
        assert result["success"] is False
        assert "No file path provided" in result["error"]
        attachments_mixin.jira.add_attachment.assert_not_called()

    def test_upload_attachment_file_not_found(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test attachment upload when file doesn't exist."""
        # Mock file operations
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = False
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"

            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is False
            assert "File not found" in result["error"]
            attachments_mixin.jira.add_attachment.assert_not_called()

    def test_upload_attachment_api_error(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload with an API error."""
        # Mock the Jira API to raise an exception
        attachments_mixin.jira.add_attachment.side_effect = Exception("API Error")

        # Mock file operations
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.basename") as mock_basename,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = True
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"
            mock_basename.return_value = "test_file.txt"

            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is False
            assert "API Error" in result["error"]

    def test_upload_attachment_no_response(self, attachments_mixin: AttachmentsMixin):
        """Test attachment upload when API returns no response."""
        # Mock the Jira API to return None
        attachments_mixin.jira.add_attachment.return_value = None

        # Mock file operations
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.isabs") as mock_isabs,
            patch("os.path.abspath") as mock_abspath,
            patch("os.path.basename") as mock_basename,
            patch("builtins.open", mock_open(read_data=b"test content")),
        ):
            mock_exists.return_value = True
            mock_isabs.return_value = True
            mock_abspath.return_value = "/absolute/path/test_file.txt"
            mock_basename.return_value = "test_file.txt"

            result = attachments_mixin.upload_attachment(
                "TEST-123", "/absolute/path/test_file.txt"
            )

            # Assertions
            assert result["success"] is False
            assert "Failed to upload attachment" in result["error"]

    # Tests for upload_attachments method

    def test_upload_attachments_success(self, attachments_mixin: AttachmentsMixin):
        """Test successful upload of multiple attachments."""
        # Set up mock for upload_attachment method to simulate successful uploads
        file_paths = [
            "/path/to/file1.txt",
            "/path/to/file2.pdf",
            "/path/to/file3.jpg",
        ]

        # Create mock successful results for each file
        mock_results = [
            {
                "success": True,
                "issue_key": "TEST-123",
                "filename": f"file{i + 1}.{ext}",
                "size": 100 * (i + 1),
                "id": f"id{i + 1}",
            }
            for i, ext in enumerate(["txt", "pdf", "jpg"])
        ]

        with patch.object(
            attachments_mixin, "upload_attachment", side_effect=mock_results
        ) as mock_upload:
            # Call the method
            result = attachments_mixin.upload_attachments("TEST-123", file_paths)

            # Assertions
            assert result["success"] is True
            assert result["issue_key"] == "TEST-123"
            assert result["total"] == 3
            assert len(result["uploaded"]) == 3
            assert len(result["failed"]) == 0

            # Check that upload_attachment was called for each file
            assert mock_upload.call_count == 3
            mock_upload.assert_any_call("TEST-123", "/path/to/file1.txt")
            mock_upload.assert_any_call("TEST-123", "/path/to/file2.pdf")
            mock_upload.assert_any_call("TEST-123", "/path/to/file3.jpg")

            # Verify uploaded files details
            assert result["uploaded"][0]["filename"] == "file1.txt"
            assert result["uploaded"][1]["filename"] == "file2.pdf"
            assert result["uploaded"][2]["filename"] == "file3.jpg"
            assert result["uploaded"][0]["size"] == 100
            assert result["uploaded"][1]["size"] == 200
            assert result["uploaded"][2]["size"] == 300
            assert result["uploaded"][0]["id"] == "id1"
            assert result["uploaded"][1]["id"] == "id2"
            assert result["uploaded"][2]["id"] == "id3"

    def test_upload_attachments_mixed_results(
        self, attachments_mixin: AttachmentsMixin
    ):
        """Test upload of multiple attachments with mixed success and failure."""
        # Set up mock for upload_attachment method to simulate mixed results
        file_paths = [
            "/path/to/file1.txt",  # Will succeed
            "/path/to/file2.pdf",  # Will fail
            "/path/to/file3.jpg",  # Will succeed
        ]

        # Create mock results with mixed success/failure
        mock_results = [
            {
                "success": True,
                "issue_key": "TEST-123",
                "filename": "file1.txt",
                "size": 100,
                "id": "id1",
            },
            {"success": False, "error": "File not found: /path/to/file2.pdf"},
            {
                "success": True,
                "issue_key": "TEST-123",
                "filename": "file3.jpg",
                "size": 300,
                "id": "id3",
            },
        ]

        with patch.object(
            attachments_mixin, "upload_attachment", side_effect=mock_results
        ) as mock_upload:
            # Call the method
            result = attachments_mixin.upload_attachments("TEST-123", file_paths)

            # Assertions
            assert (
                result["success"] is True
            )  # Overall success is True even with partial failures
            assert result["issue_key"] == "TEST-123"
            assert result["total"] == 3
            assert len(result["uploaded"]) == 2
            assert len(result["failed"]) == 1

            # Check that upload_attachment was called for each file
            assert mock_upload.call_count == 3

            # Verify uploaded files details
            assert result["uploaded"][0]["filename"] == "file1.txt"
            assert result["uploaded"][1]["filename"] == "file3.jpg"
            assert result["uploaded"][0]["size"] == 100
            assert result["uploaded"][1]["size"] == 300
            assert result["uploaded"][0]["id"] == "id1"
            assert result["uploaded"][1]["id"] == "id3"

            # Verify failed file details
            assert result["failed"][0]["filename"] == "file2.pdf"
            assert "File not found" in result["failed"][0]["error"]

    def test_upload_attachments_empty_list(self, attachments_mixin: AttachmentsMixin):
        """Test upload with an empty list of file paths."""
        # Call the method with an empty list
        result = attachments_mixin.upload_attachments("TEST-123", [])

        # Assertions
        assert result["success"] is False
        assert "No file paths provided" in result["error"]

    def test_upload_attachments_no_issue_key(self, attachments_mixin: AttachmentsMixin):
        """Test upload with no issue key provided."""
        # Call the method with no issue key
        result = attachments_mixin.upload_attachments("", ["/path/to/file.txt"])

        # Assertions
        assert result["success"] is False
        assert "No issue key provided" in result["error"]


class TestFilesystemUploadStagingBackend:
    """Tests for the shared-filesystem upload staging backend."""

    @pytest.fixture
    def backend(self, tmp_path):
        return FilesystemUploadStagingBackend(
            root_dir=str(tmp_path / "staging"),
            ttl_minutes=30,
            max_size_mb=1,
        )

    def test_is_a_backend(self, backend):
        assert isinstance(backend, UploadStagingBackend)

    def test_create_and_validate_session(self, backend):
        session_id = backend.create_session()
        assert session_id
        assert backend.is_valid_session(session_id) is True

    def test_invalid_session_id_rejected(self, backend):
        assert backend.is_valid_session("../etc/passwd") is False
        assert backend.is_valid_session("unknown") is False

    def test_store_and_get_roundtrip(self, backend):
        session_id = backend.create_session()
        file_id = backend.store(session_id, "hello.txt", b"hi there", "text/plain")
        entry = backend.get(session_id, file_id)
        assert entry is not None
        assert entry["filename"] == "hello.txt"
        assert entry["content"] == b"hi there"
        assert entry["mime_type"] == "text/plain"
        assert entry["created_at"] is not None
        assert entry["expires_at"] is not None

    def test_store_rejects_unknown_session(self, backend):
        with pytest.raises(PermissionError):
            backend.store("unknown-session", "f.txt", b"x", "text/plain")

    def test_store_rejects_oversize_file(self, backend):
        session_id = backend.create_session()
        with pytest.raises(ValueError):
            backend.store(session_id, "big.bin", b"x" * (1024 * 1024 + 1), "app/bin")

    def test_remove_deletes_file(self, backend):
        session_id = backend.create_session()
        file_id = backend.store(session_id, "f.txt", b"data", "text/plain")
        backend.remove(session_id, file_id)
        assert backend.get(session_id, file_id) is None

    def test_get_unknown_returns_none(self, backend):
        session_id = backend.create_session()
        assert backend.get(session_id, "missing") is None
        assert backend.get(session_id, "../evil") is None

    def test_expired_session_is_invalid(self, tmp_path):
        backend = FilesystemUploadStagingBackend(
            root_dir=str(tmp_path / "staging"),
            ttl_minutes=0,
            max_size_mb=1,
        )
        session_id = backend.create_session()
        assert backend.is_valid_session(session_id) is False

    def test_clear_removes_sessions(self, backend):
        session_id = backend.create_session()
        backend.store(session_id, "f.txt", b"data", "text/plain")
        backend.clear()
        assert backend.is_valid_session(session_id) is False

    def test_sweep_expired_reclaims_whole_expired_session(self, backend):
        """A never-finalized session is reclaimed once its TTL elapses."""
        session_id = backend.create_session()
        backend.store(session_id, "f.txt", b"data", "text/plain")
        future = _cache_utcnow() + timedelta(hours=1)
        with patch("mcp_atlassian.jira.upload_staging._utcnow", return_value=future):
            removed = backend.sweep_expired()
        assert removed == 1
        assert backend.get(session_id, "f.txt") is None

    def test_sweep_expired_keeps_live_session(self, backend):
        """Unexpired staged files survive a sweep."""
        session_id = backend.create_session()
        file_id = backend.store(session_id, "f.txt", b"data", "text/plain")
        removed = backend.sweep_expired()
        assert removed == 0
        assert backend.get(session_id, file_id) is not None

    def test_persists_across_instances(self, tmp_path):
        root = str(tmp_path / "shared")
        first = FilesystemUploadStagingBackend(root_dir=root, max_size_mb=1)
        session_id = first.create_session()
        file_id = first.store(session_id, "f.txt", b"shared-bytes", "text/plain")

        # A second instance (simulating a different server pod) sees the file.
        second = FilesystemUploadStagingBackend(root_dir=root, max_size_mb=1)
        entry = second.get(session_id, file_id)
        assert entry is not None
        assert entry["content"] == b"shared-bytes"

    def test_uri_helpers_inherited(self, backend):
        uri = backend.make_uri("sess", "file")
        assert uri == "upload://sessions/sess/file"
        assert backend.parse_uri(uri) == ("sess", "file")


class TestGetUploadStagingFactory:
    """Tests for the environment-driven upload staging backend factory."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        import mcp_atlassian.jira.upload_staging as staging_module

        staging_module._upload_staging = None
        yield
        staging_module._upload_staging = None

    def test_defaults_to_memory(self):
        from mcp_atlassian.jira.upload_staging import get_upload_staging

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UPLOAD_STAGING_BACKEND", None)
            backend = get_upload_staging()
        assert isinstance(backend, UploadStagingStore)

    def test_memory_explicit(self):
        from mcp_atlassian.jira.upload_staging import get_upload_staging

        with patch.dict(os.environ, {"UPLOAD_STAGING_BACKEND": "memory"}):
            backend = get_upload_staging()
        assert isinstance(backend, UploadStagingStore)

    def test_filesystem_backend(self, tmp_path):
        from mcp_atlassian.jira.upload_staging import get_upload_staging

        with patch.dict(
            os.environ,
            {
                "UPLOAD_STAGING_BACKEND": "filesystem",
                "UPLOAD_STAGING_DIR": str(tmp_path / "fs"),
            },
        ):
            backend = get_upload_staging()
        assert isinstance(backend, FilesystemUploadStagingBackend)

    def test_filesystem_requires_dir(self):
        from mcp_atlassian.jira.upload_staging import get_upload_staging

        with patch.dict(os.environ, {"UPLOAD_STAGING_BACKEND": "filesystem"}):
            os.environ.pop("UPLOAD_STAGING_DIR", None)
            with pytest.raises(ValueError, match="UPLOAD_STAGING_DIR"):
                get_upload_staging()

    def test_unknown_backend_raises(self):
        from mcp_atlassian.jira.upload_staging import get_upload_staging

        with patch.dict(os.environ, {"UPLOAD_STAGING_BACKEND": "nope"}):
            with pytest.raises(ValueError, match="Unknown UPLOAD_STAGING_BACKEND"):
                get_upload_staging()

    def test_dotted_path_backend(self):
        from mcp_atlassian.jira.upload_staging import get_upload_staging

        path = "mcp_atlassian.jira.upload_staging:FilesystemUploadStagingBackend"
        # FilesystemUploadStagingBackend needs a root_dir positional arg, so a
        # bare dotted path to it will fail construction; use a tiny stub instead.
        with patch.dict(os.environ, {"UPLOAD_STAGING_BACKEND": path}):
            with pytest.raises(TypeError):
                get_upload_staging()

    def test_invalid_dotted_path_raises(self):
        from mcp_atlassian.jira.upload_staging import get_upload_staging

        with patch.dict(
            os.environ, {"UPLOAD_STAGING_BACKEND": "not.a.real.module:Nope"}
        ):
            with pytest.raises((ImportError, ValueError, AttributeError)):
                get_upload_staging()


class TestFilesystemDownloadTokenStore:
    """Tests for the shared-filesystem download token store."""

    @pytest.fixture
    def store(self, tmp_path):
        return FilesystemDownloadTokenStore(root_dir=str(tmp_path / "downloads"))

    def _expiry(self, minutes: int = 5):
        from datetime import timedelta

        from mcp_atlassian.jira.attachment_cache import _utcnow

        return _utcnow() + timedelta(minutes=minutes)

    def test_is_a_store(self, store):
        assert isinstance(store, DownloadTokenStore)

    def test_create_and_get_roundtrip(self, store):
        token = store.create(
            "PROJ-1", "report.pdf", "application/pdf", b"pdf-bytes", self._expiry()
        )
        entry = store.get(token)
        assert entry is not None
        assert entry["content"] == b"pdf-bytes"
        assert entry["mime_type"] == "application/pdf"
        assert entry["filename"] == "report.pdf"
        assert entry["issue_key"] == "PROJ-1"

    def test_unknown_token_returns_none(self, store):
        assert store.get("does-not-exist") is None
        assert store.get("../evil") is None

    def test_expired_token_returns_none(self, store):
        token = store.create(
            "PROJ-1", "f.pdf", "application/pdf", b"x", self._expiry(minutes=-1)
        )
        assert store.get(token) is None

    def test_clear_removes_tokens(self, store):
        token = store.create("PROJ-1", "f.pdf", "application/pdf", b"x", self._expiry())
        store.clear()
        assert store.get(token) is None

    def test_token_resolvable_across_instances(self, tmp_path):
        """A token minted by one instance is served by another sharing the dir."""
        root = str(tmp_path / "shared")
        first = FilesystemDownloadTokenStore(root_dir=root)
        token = first.create(
            "PROJ-1", "f.pdf", "application/pdf", b"shared-bytes", self._expiry()
        )

        # Simulate a different pod pointed at the same shared volume.
        second = FilesystemDownloadTokenStore(root_dir=root)
        entry = second.get(token)
        assert entry is not None
        assert entry["content"] == b"shared-bytes"


class TestAttachmentCacheWithFilesystemTokens:
    """AttachmentCache download tokens work across instances via shared dir."""

    def test_download_token_resolvable_by_second_cache(self, tmp_path):
        root = str(tmp_path / "dl")
        cache_a = AttachmentCache(
            ttl_minutes=10,
            max_size_mb=1,
            token_store=FilesystemDownloadTokenStore(root_dir=root),
        )
        cache_a.store(
            issue_key="PROJ-1",
            filename="report.pdf",
            content=b"pdf-bytes",
            mime_type="application/pdf",
        )
        token_info = cache_a.create_download_token("PROJ-1", "report.pdf")

        # A second cache instance (fresh, empty local cache) resolves the token.
        cache_b = AttachmentCache(
            ttl_minutes=10,
            max_size_mb=1,
            token_store=FilesystemDownloadTokenStore(root_dir=root),
        )
        attachment = cache_b.get_by_download_token(token_info["token"])
        assert attachment is not None
        assert attachment["content"] == b"pdf-bytes"
        assert attachment["mime_type"] == "application/pdf"


class TestSweepExpired:
    """Proactive cleanup of expired download tokens and staged uploads."""

    def test_filesystem_token_store_removes_unfetched_expired_token(self, tmp_path):
        """A minted-but-never-fetched download token is reclaimed on sweep."""
        root = tmp_path / "dl"
        store = FilesystemDownloadTokenStore(root_dir=str(root))
        token = store.create(
            issue_key="PROJ-1",
            filename="report.pdf",
            mime_type="application/pdf",
            content=b"pdf-bytes",
            expires_at=_cache_utcnow() - timedelta(minutes=1),
        )
        # Nothing ever reads the token, so only a sweep can reclaim it.
        assert (root / f"{token}.bin").exists()
        assert (root / f"{token}.json").exists()

        removed = store.sweep_expired()

        assert removed == 1
        assert not (root / f"{token}.bin").exists()
        assert not (root / f"{token}.json").exists()

    def test_filesystem_token_store_keeps_live_token(self, tmp_path):
        store = FilesystemDownloadTokenStore(root_dir=str(tmp_path / "dl"))
        token = store.create(
            issue_key="PROJ-1",
            filename="report.pdf",
            mime_type="application/pdf",
            content=b"pdf-bytes",
            expires_at=_cache_utcnow() + timedelta(minutes=5),
        )
        assert store.sweep_expired() == 0
        assert store.get(token) is not None

    def test_filesystem_token_store_reaps_old_temp_files(self, tmp_path):
        root = tmp_path / "dl"
        store = FilesystemDownloadTokenStore(root_dir=str(root))
        stale = root / "abc.json.deadbeef.tmp"
        stale.write_bytes(b"partial")
        old = (_cache_utcnow() - timedelta(hours=2)).timestamp()
        os.utime(stale, (old, old))

        store.sweep_expired()

        assert not stale.exists()

    def test_memory_token_store_sweep_expired(self):
        store = MemoryDownloadTokenStore()
        token = store.create(
            issue_key="PROJ-1",
            filename="f.txt",
            mime_type="text/plain",
            content=b"x",
            expires_at=_cache_utcnow() - timedelta(minutes=1),
        )
        assert store.sweep_expired() == 1
        assert store.get(token) is None

    def test_memory_staging_sweep_expired(self):
        store = UploadStagingStore(ttl_minutes=30, max_size_mb=1)
        session_id = store.create_session()
        store.store(session_id, "f.txt", b"data", "text/plain")
        future = _cache_utcnow() + timedelta(hours=1)
        with patch("mcp_atlassian.jira.upload_staging._utcnow", return_value=future):
            removed = store.sweep_expired()
        assert removed == 1

    def test_attachment_cache_sweep_delegates_to_token_store(self, tmp_path):
        store = FilesystemDownloadTokenStore(root_dir=str(tmp_path / "dl"))
        cache = AttachmentCache(ttl_minutes=10, max_size_mb=1, token_store=store)
        store.create(
            issue_key="PROJ-1",
            filename="f.txt",
            mime_type="text/plain",
            content=b"x",
            expires_at=_cache_utcnow() - timedelta(minutes=1),
        )
        assert cache.sweep_expired() >= 1


class TestBuildDownloadTokenStoreFactory:
    """Tests for the environment-driven download token store factory."""

    def test_defaults_to_memory(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATTACHMENT_DOWNLOAD_BACKEND", None)
            store = _build_download_token_store()
        assert isinstance(store, MemoryDownloadTokenStore)

    def test_filesystem_backend(self, tmp_path):
        with patch.dict(
            os.environ,
            {
                "ATTACHMENT_DOWNLOAD_BACKEND": "filesystem",
                "ATTACHMENT_DOWNLOAD_DIR": str(tmp_path / "dl"),
            },
        ):
            store = _build_download_token_store()
        assert isinstance(store, FilesystemDownloadTokenStore)

    def test_filesystem_requires_dir(self):
        with patch.dict(os.environ, {"ATTACHMENT_DOWNLOAD_BACKEND": "filesystem"}):
            os.environ.pop("ATTACHMENT_DOWNLOAD_DIR", None)
            with pytest.raises(ValueError, match="ATTACHMENT_DOWNLOAD_DIR"):
                _build_download_token_store()

    def test_unknown_backend_raises(self):
        with patch.dict(os.environ, {"ATTACHMENT_DOWNLOAD_BACKEND": "nope"}):
            with pytest.raises(ValueError, match="Unknown ATTACHMENT_DOWNLOAD_BACKEND"):
                _build_download_token_store()

    def test_invalid_dotted_path_raises(self):
        with patch.dict(
            os.environ,
            {"ATTACHMENT_DOWNLOAD_BACKEND": "not.a.real.module:Nope"},
        ):
            with pytest.raises((ImportError, ValueError, AttributeError)):
                _build_download_token_store()
