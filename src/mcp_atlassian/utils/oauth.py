"""OAuth 2.0 utilities for Atlassian Cloud and Data Center authentication.

This module provides utilities for OAuth 2.0 authentication with Atlassian.
It handles:
- OAuth configuration for Cloud and Data Center
- Token acquisition, storage, and refresh
- Session configuration for API clients
"""

import hashlib
import json
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import keyring
import requests

from .urls import is_atlassian_cloud_url

# Configure logging
logger = logging.getLogger("mcp-atlassian.oauth")

# Cloud OAuth endpoints
CLOUD_TOKEN_URL = "https://auth.atlassian.com/oauth/token"  # noqa: S105
CLOUD_AUTHORIZE_URL = "https://auth.atlassian.com/authorize"
CLOUD_ID_URL = "https://api.atlassian.com/oauth/token/accessible-resources"

# Legacy aliases for backwards compatibility
TOKEN_URL = CLOUD_TOKEN_URL  # noqa: S105
AUTHORIZE_URL = CLOUD_AUTHORIZE_URL

# Data Center OAuth endpoint paths
DC_TOKEN_PATH = "/rest/oauth2/latest/token"  # noqa: S105
DC_AUTHORIZE_PATH = "/rest/oauth2/latest/authorize"

TOKEN_EXPIRY_MARGIN = 300  # 5 minutes in seconds
HTTP_TIMEOUT = (5, 20)
KEYRING_SERVICE_NAME = "mcp-atlassian-oauth"


@dataclass
class OAuthConfig:
    """OAuth 2.0 configuration for Atlassian Cloud and Data Center.

    This class manages the OAuth configuration and tokens. It handles:
    - Authentication configuration (client credentials)
    - Token acquisition and refreshing
    - Token storage and retrieval
    - Cloud ID identification (Cloud) or base URL routing (Data Center)
    """

    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str
    cloud_id: str | None = None
    base_url: str | None = None
    refresh_token: str | None = None
    access_token: str | None = None
    expires_at: float | None = None

    def __post_init__(self) -> None:
        """Validate Cloud and Data Center routing configuration."""
        if self.cloud_id and self.base_url:
            if is_atlassian_cloud_url(self.base_url):
                self.base_url = None
            else:
                raise ValueError(
                    "OAuthConfig cannot have both cloud_id and base_url set. "
                    "Use cloud_id for Cloud or base_url for Data Center."
                )

    @property
    def is_data_center(self) -> bool:
        """Return whether this configuration targets Data Center."""
        return bool(self.base_url) and not is_atlassian_cloud_url(self.base_url)

    @property
    def token_url(self) -> str:
        """Return the token endpoint for the configured environment."""
        if self.is_data_center and self.base_url:
            return f"{self.base_url.rstrip('/')}{DC_TOKEN_PATH}"
        return CLOUD_TOKEN_URL

    @property
    def authorize_url(self) -> str:
        """Return the authorization endpoint for the configured environment."""
        if self.is_data_center and self.base_url:
            return f"{self.base_url.rstrip('/')}{DC_AUTHORIZE_PATH}"
        return CLOUD_AUTHORIZE_URL

    @property
    def is_token_expired(self) -> bool:
        """Check if the access token is expired or will expire soon.

        Returns:
            True if the token is expired or will expire soon, False otherwise.
        """
        # If we don't have a token or expiry time, consider it expired
        if not self.access_token or not self.expires_at:
            return True

        # Consider the token expired if it will expire within the margin
        return time.time() + TOKEN_EXPIRY_MARGIN >= self.expires_at

    def get_authorization_url(self, state: str) -> str:
        """Get the authorization URL for the OAuth 2.0 flow.

        Args:
            state: Random state string for CSRF protection

        Returns:
            The authorization URL to redirect the user to.
        """
        params: dict[str, str] = {
            "client_id": self.client_id,
            "scope": self.scope,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
        }
        if not self.is_data_center:
            params["audience"] = "api.atlassian.com"
            params["prompt"] = "consent"
        return f"{self.authorize_url}?{urllib.parse.urlencode(params)}"

    def exchange_code_for_tokens(self, code: str) -> bool:
        """Exchange the authorization code for access and refresh tokens.

        Args:
            code: The authorization code from the callback

        Returns:
            True if tokens were successfully acquired, False otherwise.
        """
        try:
            payload = {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
            }

            logger.info(f"Exchanging authorization code for tokens at {self.token_url}")

            response = requests.post(self.token_url, data=payload, timeout=HTTP_TIMEOUT)

            logger.debug(f"Token exchange response status: {response.status_code}")

            if not response.ok:
                logger.error(
                    f"Token exchange failed with status {response.status_code}. Response: {response.text}"
                )
                return False

            # Parse the response
            token_data = response.json()

            # Check if required tokens are present
            if "access_token" not in token_data:
                logger.error(
                    f"Access token not found in response. Keys found: {list(token_data.keys())}"
                )
                return False

            if "refresh_token" not in token_data:
                if self.is_data_center:
                    logger.warning(
                        "No refresh_token in Data Center response; reauthorize "
                        "when the access token expires."
                    )
                else:
                    logger.error(
                        "Refresh token not found in response. Ensure 'offline_access' "
                        f"scope is included. Keys found: {list(token_data.keys())}"
                    )
                    return False

            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token")
            self.expires_at = time.time() + token_data.get("expires_in", 3600)

            if not self.is_data_center:
                self._get_cloud_id()

            # Save the tokens
            self._save_tokens()

            # Log success message with token details
            logger.info(
                "OAuth token exchange successful; access token expires in %ss.",
                token_data.get("expires_in", 3600),
            )
            if self.cloud_id:
                logger.info(f"Cloud ID successfully retrieved: {self.cloud_id}")
            elif not self.is_data_center:
                logger.warning(
                    "Cloud ID was not retrieved after token exchange. Check accessible resources."
                )
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during token exchange: {e}", exc_info=True)
            return False
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON response from token endpoint: {e}",
                exc_info=True,
            )
            logger.error(
                f"Response text that failed to parse: {response.text if 'response' in locals() else 'Response object not available'}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            return False

    def refresh_access_token(self) -> bool:
        """Refresh the access token using the refresh token.

        Returns:
            True if the token was successfully refreshed, False otherwise.
        """
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        try:
            payload = {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
            }
            if self.is_data_center:
                payload["redirect_uri"] = self.redirect_uri

            logger.debug(f"Refreshing access token at {self.token_url}...")
            response = requests.post(self.token_url, data=payload, timeout=HTTP_TIMEOUT)
            response.raise_for_status()

            # Parse the response
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # Refresh token might also be rotated
            if "refresh_token" in token_data:
                self.refresh_token = token_data["refresh_token"]
            self.expires_at = time.time() + token_data.get("expires_in", 3600)

            # Save the tokens
            self._save_tokens()

            return True
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return False

    def ensure_valid_token(self) -> bool:
        """Ensure the access token is valid, refreshing if necessary.

        Returns:
            True if the token is valid (or was refreshed successfully), False otherwise.
        """
        if not self.is_token_expired:
            return True
        return self.refresh_access_token()

    def _get_cloud_id(self) -> None:
        """Get the cloud ID for the Atlassian instance.

        This method queries the accessible resources endpoint to get the cloud ID.
        The cloud ID is needed for API calls with OAuth.
        """
        if self.is_data_center or not self.access_token:
            logger.debug("No access token available to get cloud ID")
            return

        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(CLOUD_ID_URL, headers=headers, timeout=HTTP_TIMEOUT)
            response.raise_for_status()

            resources = response.json()
            if resources and len(resources) > 0:
                # Use the first cloud site (most users have only one)
                # For users with multiple sites, they might need to specify which one to use
                self.cloud_id = resources[0]["id"]
                logger.debug(f"Found cloud ID: {self.cloud_id}")
            else:
                logger.warning("No Atlassian sites found in the response")
        except Exception as e:
            logger.error(f"Failed to get cloud ID: {e}")

    def _get_keyring_username(self) -> str:
        """Get the keyring username for storing tokens.

        The username includes routing context to avoid cross-instance collisions.

        Returns:
            A username string for keyring
        """
        if self.is_data_center and self.base_url:
            url_hash = hashlib.sha256(self.base_url.encode()).hexdigest()[:8]
            return f"oauth-{self.client_id}-dc-{url_hash}"
        if self.cloud_id:
            return f"oauth-{self.client_id}-cloud-{self.cloud_id}"
        return f"oauth-{self.client_id}"

    def _save_tokens(self) -> None:
        """Save the tokens securely using keyring for later use.

        This allows the tokens to be reused between runs without requiring
        the user to go through the authorization flow again.
        """
        try:
            username = self._get_keyring_username()

            # Store token data as JSON string in keyring
            token_data = {
                "refresh_token": self.refresh_token,
                "access_token": self.access_token,
                "expires_at": self.expires_at,
                "cloud_id": self.cloud_id,
                "base_url": self.base_url,
            }

            # Store the token data in the system keyring
            keyring.set_password(KEYRING_SERVICE_NAME, username, json.dumps(token_data))

            logger.debug(f"Saved OAuth tokens to keyring for {username}")

            # Also maintain backwards compatibility with file storage
            # for environments where keyring might not work
            self._save_tokens_to_file(token_data)

        except Exception as e:
            logger.error(f"Failed to save tokens to keyring: {e}")
            # Fall back to file storage if keyring fails
            self._save_tokens_to_file()

    def _save_tokens_to_file(self, token_data: dict | None = None) -> None:
        """Save the tokens to a file as fallback storage.

        Args:
            token_data: Optional dict with token data. If not provided,
                        will use the current object attributes.
        """
        try:
            # Create the directory if it doesn't exist
            token_dir = Path.home() / ".mcp-atlassian"
            token_dir.mkdir(exist_ok=True)
            os.chmod(token_dir, 0o700)

            # Save the tokens to a file
            storage_key = self._get_keyring_username()
            token_path = token_dir / f"{storage_key}.json"

            if token_data is None:
                token_data = {
                    "refresh_token": self.refresh_token,
                    "access_token": self.access_token,
                    "expires_at": self.expires_at,
                    "cloud_id": self.cloud_id,
                    "base_url": self.base_url,
                }

            file_descriptor = os.open(
                token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(file_descriptor, "w") as f:
                json.dump(token_data, f)
            os.chmod(token_path, 0o600)

            logger.debug(f"Saved OAuth tokens to file {token_path} (fallback storage)")
        except Exception as e:
            logger.error(f"Failed to save tokens to file: {e}")

    @staticmethod
    def load_tokens(
        client_id: str,
        cloud_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Load tokens securely from keyring.

        Args:
            client_id: The OAuth client ID

        Returns:
            Dict with the token data or empty dict if no tokens found
        """
        usernames = []
        if base_url and not is_atlassian_cloud_url(base_url):
            url_hash = hashlib.sha256(base_url.encode()).hexdigest()[:8]
            usernames.append(f"oauth-{client_id}-dc-{url_hash}")
        elif cloud_id:
            usernames.append(f"oauth-{client_id}-cloud-{cloud_id}")
        usernames.append(f"oauth-{client_id}")

        for username in usernames:
            try:
                token_json = keyring.get_password(KEYRING_SERVICE_NAME, username)
                if token_json:
                    logger.debug(f"Loaded OAuth tokens from keyring for {username}")
                    return json.loads(token_json)
            except Exception as e:
                logger.warning(
                    f"Failed to load tokens from keyring: {e}. Trying file fallback."
                )
                break

        return OAuthConfig._load_tokens_from_file(
            client_id, cloud_id=cloud_id, base_url=base_url
        )

    @staticmethod
    def _load_tokens_from_file(
        client_id: str,
        cloud_id: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Load tokens from a file as fallback.

        Args:
            client_id: The OAuth client ID

        Returns:
            Dict with the token data or empty dict if no tokens found
        """
        storage_keys = []
        if base_url and not is_atlassian_cloud_url(base_url):
            url_hash = hashlib.sha256(base_url.encode()).hexdigest()[:8]
            storage_keys.append(f"oauth-{client_id}-dc-{url_hash}")
        elif cloud_id:
            storage_keys.append(f"oauth-{client_id}-cloud-{cloud_id}")
        storage_keys.append(f"oauth-{client_id}")

        for storage_key in storage_keys:
            token_path = Path.home() / ".mcp-atlassian" / f"{storage_key}.json"
            if not token_path.exists():
                continue
            try:
                with open(token_path) as f:
                    token_data = json.load(f)
                    logger.debug(
                        f"Loaded OAuth tokens from file {token_path} "
                        "(fallback storage)"
                    )
                    return token_data
            except Exception as e:
                logger.error(f"Failed to load tokens from file: {e}")
        return {}

    @classmethod
    def from_env(
        cls,
        service_url: str | None = None,
        service_type: str | None = None,
    ) -> Optional["OAuthConfig"]:
        """Create an OAuth configuration from environment variables.

        Returns:
            OAuthConfig instance or None if OAuth is not enabled
        """
        # Check if OAuth is explicitly enabled (allows minimal config)
        oauth_enabled = os.getenv("ATLASSIAN_OAUTH_ENABLE", "").lower() in (
            "true",
            "1",
            "yes",
        )

        prefix = service_type.upper() if service_type else None
        client_id = (
            os.getenv(f"{prefix}_OAUTH_CLIENT_ID") if prefix else None
        ) or os.getenv("ATLASSIAN_OAUTH_CLIENT_ID")
        client_secret = (
            os.getenv(f"{prefix}_OAUTH_CLIENT_SECRET") if prefix else None
        ) or os.getenv("ATLASSIAN_OAUTH_CLIENT_SECRET")
        redirect_uri = (
            os.getenv(f"{prefix}_OAUTH_REDIRECT_URI") if prefix else None
        ) or os.getenv("ATLASSIAN_OAUTH_REDIRECT_URI")
        scope = (os.getenv(f"{prefix}_OAUTH_SCOPE") if prefix else None) or os.getenv(
            "ATLASSIAN_OAUTH_SCOPE"
        )

        is_data_center = bool(service_url) and not is_atlassian_cloud_url(service_url)
        if is_data_center:
            redirect_uri = redirect_uri or "http://localhost:8080/callback"
            scope = scope or (
                "REPO_READ" if service_type == "bitbucket" else "WRITE"
            )

        # Full OAuth configuration (traditional mode)
        if client_id and client_secret:
            if not is_data_center and not all([redirect_uri, scope]):
                return None

            cloud_id = (
                os.getenv("ATLASSIAN_OAUTH_CLOUD_ID") if not is_data_center else None
            )
            base_url = service_url if is_data_center else None
            # Create the OAuth configuration with full credentials
            config = cls(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri or "",
                scope=scope or "",
                cloud_id=cloud_id,
                base_url=base_url,
            )

            # Try to load existing tokens
            token_data = cls.load_tokens(
                client_id, cloud_id=cloud_id, base_url=base_url
            )
            if token_data:
                config.refresh_token = token_data.get("refresh_token")
                config.access_token = token_data.get("access_token")
                config.expires_at = token_data.get("expires_at")
                if not config.cloud_id and "cloud_id" in token_data:
                    config.cloud_id = token_data["cloud_id"]
                if not config.base_url and "base_url" in token_data:
                    config.base_url = token_data["base_url"]

            return config

        # Minimal OAuth configuration (user-provided tokens mode)
        elif oauth_enabled:
            # Create minimal config that works with user-provided tokens
            logger.info(
                "Creating minimal OAuth config for user-provided tokens (ATLASSIAN_OAUTH_ENABLE=true)"
            )
            return cls(
                client_id="",  # Will be provided by user tokens
                client_secret="",  # Not needed for user tokens
                redirect_uri="",  # Not needed for user tokens
                scope="",  # Will be determined by user token permissions
                cloud_id=(
                    os.getenv("ATLASSIAN_OAUTH_CLOUD_ID")
                    if not is_data_center
                    else None
                ),
                base_url=service_url if is_data_center else None,
            )

        # No OAuth configuration
        return None


@dataclass
class BYOAccessTokenOAuthConfig:
    """OAuth configuration when providing a pre-existing access token.

    This class accepts a Cloud ID or Data Center base URL with an access token.

    This configuration does not support token refreshing.
    """

    access_token: str
    cloud_id: str | None = None
    base_url: str | None = None
    refresh_token: None = field(default=None, repr=False)
    expires_at: None = field(default=None, repr=False)

    @property
    def is_data_center(self) -> bool:
        """Return whether this configuration targets Data Center."""
        return bool(self.base_url) and not is_atlassian_cloud_url(self.base_url)

    @classmethod
    def from_env(
        cls,
        service_url: str | None = None,
        service_type: str | None = None,
    ) -> Optional["BYOAccessTokenOAuthConfig"]:
        """Create a BYOAccessTokenOAuthConfig from environment variables.

        Reads `ATLASSIAN_OAUTH_CLOUD_ID` and `ATLASSIAN_OAUTH_ACCESS_TOKEN`.

        Returns:
            BYOAccessTokenOAuthConfig instance or None if required
            environment variables are missing.
        """
        cloud_id = os.getenv("ATLASSIAN_OAUTH_CLOUD_ID")
        prefix = service_type.upper() if service_type else None
        access_token = (
            os.getenv(f"{prefix}_OAUTH_ACCESS_TOKEN") if prefix else None
        ) or os.getenv("ATLASSIAN_OAUTH_ACCESS_TOKEN")

        if not access_token:
            return None

        is_data_center = bool(service_url) and not is_atlassian_cloud_url(service_url)
        base_url = service_url if is_data_center else None
        if not cloud_id and not base_url:
            return None

        return cls(
            access_token=access_token,
            cloud_id=cloud_id if not is_data_center else None,
            base_url=base_url,
        )


def get_oauth_config_from_env(
    service_url: str | None = None,
    service_type: str | None = None,
) -> OAuthConfig | BYOAccessTokenOAuthConfig | None:
    """Get the appropriate OAuth configuration from environment variables.

    This function attempts to load standard OAuth configuration first (OAuthConfig).
    If that's not available, it tries to load a "Bring Your Own Access Token"
    configuration (BYOAccessTokenOAuthConfig).

    Returns:
        An instance of OAuthConfig or BYOAccessTokenOAuthConfig if environment
        variables are set for either, otherwise None.
    """
    return BYOAccessTokenOAuthConfig.from_env(
        service_url=service_url, service_type=service_type
    ) or OAuthConfig.from_env(service_url=service_url, service_type=service_type)


def configure_oauth_session(
    session: requests.Session, oauth_config: OAuthConfig | BYOAccessTokenOAuthConfig
) -> bool:
    """Configure a requests session with OAuth 2.0 authentication.

    This function ensures the access token is valid and adds it to the session headers.

    Args:
        session: The requests session to configure
        oauth_config: The OAuth configuration to use

    Returns:
        True if the session was successfully configured, False otherwise
    """
    logger.debug(
        f"configure_oauth_session: Received OAuthConfig with "
        f"access_token_present={bool(oauth_config.access_token)}, "
        f"refresh_token_present={bool(oauth_config.refresh_token)}, "
        f"cloud_id='{oauth_config.cloud_id}'"
    )
    if not oauth_config.access_token and not oauth_config.refresh_token:
        logger.warning(
            "configure_oauth_session: No access_token or refresh_token available."
        )
        return False
    # If user provided only an access token (no refresh_token), use it directly
    if oauth_config.access_token and not oauth_config.refresh_token:
        logger.info(
            "configure_oauth_session: Using provided OAuth access token directly (no refresh_token)."
        )
        session.headers["Authorization"] = f"Bearer {oauth_config.access_token}"
        return True
    logger.debug("configure_oauth_session: Proceeding to ensure_valid_token.")
    # Otherwise, ensure we have a valid token (refresh if needed)
    if isinstance(oauth_config, BYOAccessTokenOAuthConfig):
        logger.error(
            "configure_oauth_session: oauth access token configuration provided as empty string."
        )
        return False
    if not oauth_config.ensure_valid_token():
        logger.error(
            f"configure_oauth_session: ensure_valid_token returned False. "
            f"Token was expired: {oauth_config.is_token_expired}, "
            f"Refresh token present for attempt: {bool(oauth_config.refresh_token)}"
        )
        return False
    session.headers["Authorization"] = f"Bearer {oauth_config.access_token}"
    logger.info("Successfully configured OAuth session for Atlassian API")
    return True
