"""Read model limits and pricing without importing an LLM client SDK."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.resources import files

import httpx

from portia.logger import logger

MODEL_CATALOG_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
_FIELDS = ("max_input_tokens", "input_cost_per_token", "output_cost_per_token")
_RESERVED_KEYS = {"sample_spec", "fallback_generalizations"}


def _metadata(entry: object) -> dict[str, int | float]:
    if not isinstance(entry, dict):
        raise TypeError("Model catalog entry must be an object")
    metadata = {}
    for field in _FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Model catalog value must be a finite non-negative number")
        if field == "max_input_tokens" and not isinstance(value, int):
            raise TypeError("Model context window must be an integer")
        metadata[field] = value
    return metadata


def _parse_catalog(payload: object) -> dict[str, dict[str, int | float]]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Model catalog must be a non-empty object")
    catalog = {}
    aliases = {}
    for name, entry in payload.items():
        if name in _RESERVED_KEYS:
            continue
        metadata = _metadata(entry)
        catalog[name] = metadata
        model_aliases = entry.get("aliases", [])
        if not isinstance(model_aliases, list) or any(
            not isinstance(alias, str) for alias in model_aliases
        ):
            raise ValueError("Model catalog aliases must be a list of names")
        for alias in model_aliases:
            aliases.setdefault(alias, metadata)
    if not catalog or not any(catalog.values()):
        raise ValueError("Model catalog has no metadata")
    return aliases | catalog


@lru_cache(maxsize=1)
def get_model_catalog() -> dict[str, dict[str, int | float]]:
    """Fetch once per process, falling back to the bundled catalog on failure."""
    try:
        response = httpx.get(MODEL_CATALOG_URL, timeout=5.0)
        response.raise_for_status()
        return _parse_catalog(response.json())
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger().warning(
            "Model catalog refresh failed ({}); using bundled model metadata",
            type(error).__name__,
        )
    return _parse_catalog(
        json.loads(
            files("portia").joinpath("data/model_catalog.json").read_text(encoding="utf-8"),
        )
    )
