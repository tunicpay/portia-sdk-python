"""Model metadata remains available without the LiteLLM SDK or a live catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import httpx
import pytest

from portia import model_catalog
from portia.cost_estimator import CostEstimator
from portia.model import LangChainGenerativeModel
from tests.utils import get_mock_base_chat_model

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def clear_catalog_cache() -> Iterator[None]:
    """Keep each test independent of the process-wide catalog cache."""
    model_catalog.get_model_catalog.cache_clear()
    yield
    model_catalog.get_model_catalog.cache_clear()


def test_remote_catalog_is_cached_and_aliases_preserve_canonical_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use current prices once, including aliases, without overwriting named models."""
    response = httpx.Response(
        200,
        request=httpx.Request("GET", model_catalog.MODEL_CATALOG_URL),
        json={
            "current": {
                "max_input_tokens": 200000,
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "aliases": ["alias", "canonical"],
            },
            "canonical": {"max_input_tokens": 100000},
        },
    )
    fetch = Mock(return_value=response)
    monkeypatch.setattr(model_catalog.httpx, "get", fetch)
    catalog = model_catalog.get_model_catalog()
    assert catalog["alias"] == catalog["current"]
    assert catalog["canonical"]["max_input_tokens"] == 100000
    assert catalog["current"]["input_cost_per_token"] == 0.000003
    assert model_catalog.get_model_catalog() == catalog
    fetch.assert_called_once_with(model_catalog.MODEL_CATALOG_URL, timeout=5.0)


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("unavailable"),
        httpx.ReadTimeout("timed out"),
        httpx.HTTPStatusError(
            "HTTP failure",
            request=httpx.Request("GET", "https://example.com"),
            response=httpx.Response(503),
        ),
        ValueError("invalid JSON"),
    ],
)
def test_catalog_failure_preserves_known_model_values_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    """A failed refresh must retain the locked catalog's token limits and prices."""
    monkeypatch.setattr(model_catalog.httpx, "get", Mock(side_effect=failure))
    log = Mock()
    monkeypatch.setattr(model_catalog, "logger", Mock(return_value=log))
    catalog = model_catalog.get_model_catalog()
    assert catalog["gemini-2.5-flash"] == {
        "max_input_tokens": 1048576,
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000025,
    }
    log.warning.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"model": None},
        {"model": {"max_input_tokens": "large"}},
        {"model": {"input_cost_per_token": -1}},
        {"model": {"output_cost_per_token": float("nan")}},
    ],
)
def test_malformed_catalog_uses_bundled_values(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    """Invalid upstream data must not replace usable metadata."""
    response = Mock()
    response.json.return_value = payload
    monkeypatch.setattr(model_catalog.httpx, "get", Mock(return_value=response))
    assert model_catalog.get_model_catalog()["gpt-4o"]["max_input_tokens"] == 128000


def test_portia_lookups_preserve_limits_prices_and_unknown_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Portia consumes the same metadata and keeps its existing unknown-model defaults."""
    monkeypatch.setattr(model_catalog.httpx, "get", Mock(side_effect=httpx.ConnectError("offline")))
    client = get_mock_base_chat_model()
    model = LangChainGenerativeModel(client=client, model_name="gemini-2.5-flash")
    assert model.get_context_window_size() == 1048576
    unknown = LangChainGenerativeModel(client=client, model_name="unknown-model")
    assert unknown.get_context_window_size() == 100000
    estimator = object.__new__(CostEstimator)
    assert estimator._get_model_pricing("gemini-2.5-flash") == {"input": 0.3, "output": 2.5}
    assert estimator._calculate_cost(1000000, 1000000, "gemini-2.5-flash") == 2.8
    assert estimator._get_model_pricing("unknown-model") == {"input": 5.0, "output": 15.0}
