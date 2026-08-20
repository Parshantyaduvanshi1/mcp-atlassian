"""Token verifier used for Atlassian opaque OAuth access tokens."""

from __future__ import annotations

import hashlib
import time
from urllib.parse import urlparse

import httpx
from cachetools import TTLCache
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from mcp_atlassian.utils.oauth import CLOUD_ID_URL
from mcp_atlassian.utils.urls import is_atlassian_cloud_url


class AtlassianOpaqueTokenVerifier(TokenVerifier):
    """Validate Atlassian tokens and add tenant routing information."""

    def __init__(
        self,
        instance_url: str,
        required_scopes: list[str] | None = None,
    ) -> None:
        super().__init__(required_scopes=required_scopes)
        self.instance_url = instance_url
        self._cache: TTLCache[str, AccessToken] = TTLCache(maxsize=256, ttl=300)

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        cached = self._cache.get(token_hash)
        if cached:
            return cached

        claims: dict[str, str] = {}
        if is_atlassian_cloud_url(self.instance_url):
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    response = await client.get(
                        CLOUD_ID_URL,
                        headers={"Authorization": f"Bearer {token}"},
                    )
                response.raise_for_status()
                resources = response.json()
            except (httpx.HTTPError, ValueError):
                return None

            if not isinstance(resources, list) or not resources:
                return None
            instance_host = (urlparse(self.instance_url).hostname or "").lower()
            resource = next(
                (
                    item
                    for item in resources
                    if isinstance(item, dict)
                    and (
                        urlparse(str(item.get("url", ""))).hostname or ""
                    ).lower()
                    == instance_host
                ),
                resources[0],
            )
            if not isinstance(resource, dict) or not resource.get("id"):
                return None
            claims["cloud_id"] = str(resource["id"])
        else:
            claims["base_url"] = self.instance_url.rstrip("/")

        access_token = AccessToken(
            token=token,
            client_id="atlassian",
            scopes=self.required_scopes or [],
            expires_at=int(time.time()) + 86400 * 30,
            claims=claims,
        )
        self._cache[token_hash] = access_token
        return access_token