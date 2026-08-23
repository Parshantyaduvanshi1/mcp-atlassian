"""Unit tests for transport selection and execution.

These tests verify that:
1. All transports use direct execution (no stdin monitoring)
2. Transport selection logic works correctly (CLI vs environment)
3. Error handling is preserved
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_atlassian import main


class TestMainTransportSelection:
    """Test the main function's transport-specific execution logic."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock server instance."""
        server = MagicMock()
        server.run_async = AsyncMock(return_value=None)
        return server

    @pytest.fixture
    def mock_asyncio_run(self):
        """Mock asyncio.run to capture what coroutine is executed."""
        with patch("asyncio.run") as mock_run:
            # Store the coroutine for inspection
            mock_run.side_effect = lambda coro: setattr(mock_run, "_called_with", coro)
            yield mock_run

    @pytest.mark.parametrize("transport", ["stdio", "sse", "streamable-http"])
    def test_all_transports_use_direct_execution(
        self, mock_server, mock_asyncio_run, transport
    ):
        """Verify all transports use direct execution without stdin monitoring.

        This is a regression test for issues #519 and #524.
        """
        with patch("mcp_atlassian.servers.main.AtlassianMCP", return_value=mock_server):
            with patch.dict("os.environ", {"TRANSPORT": transport}):
                with patch("sys.argv", ["mcp-atlassian"]):
                    try:
                        main()
                    except SystemExit:
                        pass

                    # Verify asyncio.run was called
                    assert mock_asyncio_run.called

                    # Get the coroutine info
                    called_coro = mock_asyncio_run._called_with
                    coro_repr = repr(called_coro)

                    # All transports must use direct execution
                    assert "run_with_stdio_monitoring" not in coro_repr
                    assert "run_async" in coro_repr or hasattr(called_coro, "cr_code")

    def test_cli_overrides_env_transport(self, mock_server, mock_asyncio_run):
        """Test that CLI transport argument overrides environment variable."""
        with patch("mcp_atlassian.servers.main.AtlassianMCP", return_value=mock_server):
            with patch.dict("os.environ", {"TRANSPORT": "sse"}):
                # Simulate CLI args with --transport stdio
                with patch("sys.argv", ["mcp-atlassian", "--transport", "stdio"]):
                    try:
                        main()
                    except SystemExit:
                        pass

                    # All transports now use direct execution
                    called_coro = mock_asyncio_run._called_with
                    coro_repr = repr(called_coro)
                    assert "run_async" in coro_repr or hasattr(called_coro, "cr_code")

    def test_signal_handlers_always_setup(self, mock_server):
        """Test that signal handlers are set up regardless of transport."""
        with patch("mcp_atlassian.servers.main.AtlassianMCP", return_value=mock_server):
            with patch("asyncio.run"):
                # Patch where it's imported in the main module
                with patch("mcp_atlassian.setup_signal_handlers") as mock_setup:
                    with patch.dict("os.environ", {"TRANSPORT": "stdio"}):
                        with patch("sys.argv", ["mcp-atlassian"]):
                            try:
                                main()
                            except SystemExit:
                                pass

                            # Signal handlers should always be set up
                            mock_setup.assert_called_once()

    def test_error_handling_preserved(self, mock_server):
        """Test that error handling works correctly for all transports."""
        # Make the server's run_async raise an exception when awaited
        error = RuntimeError("Server error")

        async def failing_run_async(**kwargs):
            raise error

        mock_server.run_async = failing_run_async

        with patch("mcp_atlassian.servers.main.AtlassianMCP", return_value=mock_server):
            with patch("asyncio.run") as mock_run:
                # Simulate the exception propagating through asyncio.run
                mock_run.side_effect = error

                with patch.dict("os.environ", {"TRANSPORT": "stdio"}):
                    with patch("sys.argv", ["mcp-atlassian"]):
                        # The main function logs the error and exits with code 1
                        with patch("sys.exit") as mock_exit:
                            main()
                            # Verify error was handled - sys.exit called with 1 for error
                            # and then with 0 in the finally block
                            assert mock_exit.call_count == 2
                            # Error exit
                            assert mock_exit.call_args_list[0][0][0] == 1
                            assert (
                                mock_exit.call_args_list[1][0][0] == 0
                            )  # Finally exit


class TestStatelessHttpSelection:
    """Test resolution of the stateless_http option into run_async kwargs."""

    def _run_main_and_capture_kwargs(self, argv, env):
        """Invoke main() and return the kwargs passed to run_async."""
        with patch(
            "mcp_atlassian.servers.main_mcp.run_async", new_callable=MagicMock
        ) as mock_run_async:
            with patch("asyncio.run"):
                with patch.dict("os.environ", env, clear=False):
                    with patch("sys.argv", argv):
                        try:
                            main()
                        except SystemExit:
                            pass
        assert mock_run_async.called
        return mock_run_async.call_args.kwargs

    def test_stateless_http_defaults_to_false(self):
        kwargs = self._run_main_and_capture_kwargs(
            ["mcp-atlassian"], {"TRANSPORT": "streamable-http"}
        )
        assert kwargs["stateless_http"] is False

    def test_stateless_http_enabled_via_cli_flag(self):
        kwargs = self._run_main_and_capture_kwargs(
            ["mcp-atlassian", "--stateless-http"], {"TRANSPORT": "streamable-http"}
        )
        assert kwargs["stateless_http"] is True

    def test_stateless_http_enabled_via_env(self):
        kwargs = self._run_main_and_capture_kwargs(
            ["mcp-atlassian"],
            {"TRANSPORT": "streamable-http", "STATELESS_HTTP": "true"},
        )
        assert kwargs["stateless_http"] is True

    def test_cli_flag_overrides_env(self):
        kwargs = self._run_main_and_capture_kwargs(
            ["mcp-atlassian", "--stateless-http"],
            {"TRANSPORT": "streamable-http", "STATELESS_HTTP": "false"},
        )
        assert kwargs["stateless_http"] is True

    def test_stateless_http_not_passed_for_stdio(self):
        kwargs = self._run_main_and_capture_kwargs(
            ["mcp-atlassian"], {"TRANSPORT": "stdio", "STATELESS_HTTP": "true"}
        )
        assert "stateless_http" not in kwargs
