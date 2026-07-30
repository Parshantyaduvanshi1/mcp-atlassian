"""Tests for Prometheus metrics collection utilities."""

import http
from enum import IntEnum

import pytest

from mcp_atlassian.utils import prometheus_metrics as pm
from mcp_atlassian.utils.prometheus_metrics import PROMETHEUS_AVAILABLE

pytestmark = pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE, reason="prometheus_client is not installed"
)


@pytest.fixture(scope="module")
def metrics():
    """Return the shared metrics singleton with live counters.

    ``initialize_metrics`` caches a single instance, which avoids the duplicate
    Prometheus registration that would otherwise leave a second instance's
    counters set to ``None``.
    """
    instance = pm.initialize_metrics(pod_name="test-pod")
    if instance.http_requests is None:
        pytest.skip("http_requests counter is not available")
    return instance


def _status_code_label_for(endpoint: str) -> str | None:
    """Return the status_code label recorded for a given endpoint, if any."""
    from prometheus_client import REGISTRY

    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if (
                sample.name == "mcp_atlassian_http_requests_total"
                and sample.labels.get("endpoint") == endpoint
                and sample.value > 0
            ):
                return sample.labels.get("status_code")
    return None


class TestEndRequestTrackingStatusCode:
    """Regression tests for status_code label normalization."""

    def test_httpstatus_enum_is_normalized_to_numeric_string(self, metrics):
        """HTTPStatus enum values must be recorded as numeric strings.

        ``str(http.HTTPStatus.OK)`` returns ``"HTTPStatus.OK"`` on Python 3.10,
        which would corrupt the ``status_code`` label. The metric must record
        ``"200"`` regardless of whether an int or an HTTPStatus is passed.
        """
        endpoint = "/regression-enum"
        context = metrics.start_request_tracking(method="GET", endpoint=endpoint)
        metrics.end_request_tracking(context, status_code=http.HTTPStatus.OK)

        assert _status_code_label_for(endpoint) == "200"

    def test_int_status_code_is_recorded_as_numeric_string(self, metrics):
        """Plain int status codes must still be recorded as numeric strings."""
        endpoint = "/regression-int"
        context = metrics.start_request_tracking(method="GET", endpoint=endpoint)
        metrics.end_request_tracking(context, status_code=404)

        assert _status_code_label_for(endpoint) == "404"

    def test_name_returning_intenum_is_coerced_to_numeric_string(self, metrics):
        """An IntEnum whose ``str()`` is non-numeric must still be coerced.

        Python 3.10's ``http.HTTPStatus`` stringifies to ``"HTTPStatus.OK"``.
        This synthetic enum reproduces that behaviour on every interpreter so
        the ``str(int(...))`` normalization is guarded regardless of the Python
        version running the suite.
        """

        class _NamedStatus(IntEnum):
            OK = 200

            def __str__(self) -> str:  # pragma: no cover - trivial
                return f"_NamedStatus.{self.name}"

        assert str(_NamedStatus.OK) != "200"

        endpoint = "/regression-named-enum"
        context = metrics.start_request_tracking(method="GET", endpoint=endpoint)
        metrics.end_request_tracking(context, status_code=_NamedStatus.OK)

        assert _status_code_label_for(endpoint) == "200"
