"""Tests for browser-based OAuth proxy configuration."""

from __future__ import annotations

from starlette.testclient import TestClient

from mcp_atlassian.servers.main import AtlassianMCP, _build_auth_provider
from mcp_atlassian.utils.oauth import CLOUD_AUTHORIZE_URL, CLOUD_TOKEN_URL


def _set_oauth_env(monkeypatch) -> None:
    monkeypatch.setenv("ATLASSIAN_OAUTH_PROXY_ENABLE", "true")
    monkeypatch.setenv("JIRA_URL", "https://acme.atlassian.net")
    monkeypatch.setenv("ATLASSIAN_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("ATLASSIAN_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv(
        "ATLASSIAN_OAUTH_REDIRECT_URI",
        "https://mcp.example.com/oauth/callback",
    )
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://mcp.example.com")
    monkeypatch.setenv(
        "ATLASSIAN_OAUTH_SCOPE", "read:jira-work offline_access"
    )


def test_build_auth_provider_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ATLASSIAN_OAUTH_PROXY_ENABLE", raising=False)

    assert _build_auth_provider() is None


def test_build_auth_provider_fails_closed_when_configuration_is_missing(
    monkeypatch,
):
    monkeypatch.setenv("ATLASSIAN_OAUTH_PROXY_ENABLE", "true")
    for name in (
        "ATLASSIAN_OAUTH_INSTANCE_URL",
        "JIRA_URL",
        "CONFLUENCE_URL",
        "ATLASSIAN_OAUTH_CLIENT_ID",
        "JIRA_OAUTH_CLIENT_ID",
        "CONFLUENCE_OAUTH_CLIENT_ID",
        "ATLASSIAN_OAUTH_CLIENT_SECRET",
        "JIRA_OAUTH_CLIENT_SECRET",
        "CONFLUENCE_OAUTH_CLIENT_SECRET",
        "ATLASSIAN_OAUTH_REDIRECT_URI",
    ):
        monkeypatch.delenv(name, raising=False)

    try:
        _build_auth_provider()
    except ValueError as exc:
        assert "required configuration is missing" in str(exc)
    else:
        raise AssertionError("OAuth mode must not start without its configuration")


def test_build_auth_provider_exposes_cloud_oauth_routes(monkeypatch):
    _set_oauth_env(monkeypatch)

    provider = _build_auth_provider()

    assert provider is not None
    assert provider._upstream_authorization_endpoint == CLOUD_AUTHORIZE_URL
    assert provider._upstream_token_endpoint == CLOUD_TOKEN_URL
    route_paths = {route.path for route in provider.get_routes("/mcp")}
    assert "/authorize" in route_paths
    assert "/token" in route_paths
    assert "/register" in route_paths
    assert "/.well-known/oauth-authorization-server" in route_paths
    assert "/.well-known/oauth-protected-resource/mcp" in route_paths
    assert "/oauth/callback" in route_paths


def test_oauth_mode_challenges_request_without_auth_header(monkeypatch):
    _set_oauth_env(monkeypatch)
    provider = _build_auth_provider()
    assert provider is not None
    server = AtlassianMCP(name="OAuth Test", auth=provider)
    app = server.http_app(path="/mcp", transport="streamable-http")

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert "resource_metadata=" in response.headers["www-authenticate"]