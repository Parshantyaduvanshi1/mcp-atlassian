"""Token verifier used for Atlassian opaque OAuth access tokens."""

from __future__ import annotations

import hashlib
import time
from typing import Literal

import httpx
from cachetools import TTLCache
from fastmcp.server.auth.auth import AccessToken, TokenVerifier

from mcp_atlassian.utils.urls import is_atlassian_cloud_url


class AtlassianDataCenterTokenVerifier(TokenVerifier):
    """Validate Data Center OAuth tokens against their product instance."""

    def __init__(
        self,
        instance_url: str,
        product: Literal["jira", "confluence", "bitbucket"],
        required_scopes: list[str] | None = None,
    ) -> None:
        super().__init__(required_scopes=required_scopes)
        if is_atlassian_cloud_url(instance_url):
            raise ValueError(
                "AtlassianDataCenterTokenVerifier does not support Cloud URLs"
            )
        self.instance_url = instance_url
        self.product = product
        self._cache: TTLCache[str, AccessToken] = TTLCache(maxsize=256, ttl=300)

    async def _validate_data_center_token(self, token: str) -> bool:
        """Validate a token against a lightweight authenticated product API."""
        validation_paths = {
            "jira": "/rest/api/2/myself",
            "confluence": "/rest/api/user/current",
            "bitbucket": "/rest/api/1.0/projects?limit=1",
        }
        if self.product not in validation_paths:
            return False

        validation_url = (
            f"{self.instance_url.rstrip('/')}{validation_paths[self.product]}"
        )
        try:
            async with httpx.AsyncClient(
                timeout=20,
                follow_redirects=False,
            ) as client:
                response = await client.get(
                    validation_url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                )
            response.raise_for_status()
            return isinstance(response.json(), dict)
        except (httpx.HTTPError, ValueError):
            return False

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        cached = self._cache.get(token_hash)
        if cached:
            return cached

        if not await self._validate_data_center_token(token):
            return None

        access_token = AccessToken(
            token=token,
            client_id="atlassian",
            scopes=self.required_scopes or [],
            expires_at=int(time.time()) + 86400 * 30,
            claims={"base_url": self.instance_url.rstrip("/")},
        )
        self._cache[token_hash] = access_token
        return access_token