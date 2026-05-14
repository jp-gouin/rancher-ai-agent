"""PII detection for Kubernetes Secret payloads returned by MCP tools.

The detector walks a JSON tool result, finds objects with ``kind == "Secret"``,
and emits ``PIIMatch`` entries pointing at the values inside their ``data`` and
``stringData`` blocks. Only Secret-typed objects are touched — ConfigMap and
other resources that also use a ``data`` field pass through unredacted.
"""

import json
from collections.abc import Iterator
from typing import Any

from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware._redaction import PIIMatch

PII_TYPE = "kubernetes_secret"


def _walk_for_secret_values(node: Any) -> Iterator[str]:
    """Yield each value-string from data/stringData of every Secret found in node.

    data/stringData are flat key→string maps, so once we hit a Secret we harvest
    its values directly and stop recursing. Lists and JSON-encoded string
    envelopes (e.g. MCP's ``[{"type":"text","text":"<inner json>"}]``) are
    traversed to find Secrets nested inside them.
    """
    if isinstance(node, dict):
        if node.get("kind") == "Secret":
            for key, block in node.items():
                if key in ("data", "stringData") and isinstance(block, dict):
                    yield from (v for v in block.values() if isinstance(v, str) and v)
            return
        for value in node.values():
            yield from _walk_for_secret_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_for_secret_values(item)
    elif isinstance(node, str):
        try:
            inner = json.loads(node)
        except (json.JSONDecodeError, ValueError):
            return
        yield from _walk_for_secret_values(inner)


def detect_kubernetes_secret(text: str) -> list[PIIMatch]:
    """Detect Kubernetes Secret data values in ``text`` and return their spans.

    Returns an empty list when ``text`` is not parseable JSON or contains no
    Secret-typed objects. The cursor advances after each match, guaranteeing
    non-overlapping spans even when the same value occurs more than once.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []

    matches: list[PIIMatch] = []
    cursor = 0
    for value in _walk_for_secret_values(parsed):
        start = text.find(value, cursor)
        if start < 0:
            continue
        end = start + len(value)
        matches.append(PIIMatch(type=PII_TYPE, value=value, start=start, end=end))
        cursor = end
    return matches


def create_k8s_secret_pii_middleware() -> PIIMiddleware:
    """Return a PIIMiddleware configured to redact K8s Secret data in tool results."""
    return PIIMiddleware(
        PII_TYPE,
        detector=detect_kubernetes_secret,
        strategy="redact",
        apply_to_input=False,
        apply_to_output=False,
        apply_to_tool_results=True,
    )
