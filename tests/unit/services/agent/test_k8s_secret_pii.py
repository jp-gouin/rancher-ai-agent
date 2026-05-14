"""Unit tests for the k8s_secret_pii detector and middleware factory."""

import json

from langchain.agents.middleware._redaction import apply_strategy

from app.services.agent.k8s_secret_pii import (
    PII_TYPE,
    create_k8s_secret_pii_middleware,
    detect_kubernetes_secret,
)

REDACTED = f"[REDACTED_{PII_TYPE.upper()}]"


def _redact(text: str) -> str:
    """Run the detector and apply the redact strategy, mimicking PIIMiddleware."""
    return apply_strategy(text, detect_kubernetes_secret(text), "redact")


# ============================================================================
# detect_kubernetes_secret
# ============================================================================


def test_detector_redacts_secret_data():
    """data values inside a Secret are matched and redacted."""
    text = json.dumps({
        "kind": "Secret",
        "metadata": {"name": "db"},
        "data": {"username": "YWRtaW4=", "password": "c2VjcmV0"},
    })

    matches = detect_kubernetes_secret(text)

    assert {m["value"] for m in matches} == {"YWRtaW4=", "c2VjcmV0"}
    assert all(m["type"] == PII_TYPE for m in matches)

    redacted = _redact(text)
    assert "YWRtaW4=" not in redacted
    assert "c2VjcmV0" not in redacted
    assert redacted.count(REDACTED) == 2


def test_detector_redacts_string_data():
    """stringData values inside a Secret are matched and redacted."""
    text = json.dumps({
        "kind": "Secret",
        "stringData": {"token": "plaintext-token-value"},
    })

    matches = detect_kubernetes_secret(text)

    assert len(matches) == 1
    assert matches[0]["value"] == "plaintext-token-value"

    redacted = _redact(text)
    assert "plaintext-token-value" not in redacted
    assert REDACTED in redacted


def test_detector_skips_configmap():
    """ConfigMap data is not redacted — only Secret kinds are sensitive."""
    text = json.dumps({
        "kind": "ConfigMap",
        "metadata": {"name": "app-config"},
        "data": {"key": "public-value"},
    })

    matches = detect_kubernetes_secret(text)

    assert matches == []
    assert _redact(text) == text


def test_detector_handles_secret_list():
    """Items inside a List of Secrets are all redacted."""
    text = json.dumps({
        "kind": "SecretList",
        "items": [
            {"kind": "Secret", "data": {"a": "Zm9v"}},
            {"kind": "Secret", "data": {"b": "YmFy"}},
        ],
    })

    matches = detect_kubernetes_secret(text)
    values = sorted(m["value"] for m in matches)

    assert values == ["Zm9v", "YmFy"] or values == sorted(["Zm9v", "YmFy"])

    redacted = _redact(text)
    assert "Zm9v" not in redacted
    assert "YmFy" not in redacted


def test_detector_handles_mcp_text_envelope():
    """MCP envelope [{"type":"text","text":"<json>"}] is unwrapped and matches still point to outer text."""
    inner = json.dumps({"kind": "Secret", "data": {"k": "ZW52ZWxvcGVk"}})
    text = json.dumps([{"type": "text", "text": inner}])

    matches = detect_kubernetes_secret(text)

    assert len(matches) == 1
    assert matches[0]["value"] == "ZW52ZWxvcGVk"
    # Span must point into the OUTER text (so apply_strategy can slice correctly).
    assert text[matches[0]["start"]:matches[0]["end"]] == "ZW52ZWxvcGVk"

    redacted = _redact(text)
    assert "ZW52ZWxvcGVk" not in redacted


def test_detector_invalid_json_returns_empty():
    """Non-JSON input returns no matches."""
    assert detect_kubernetes_secret("not json at all") == []
    assert detect_kubernetes_secret("") == []


def test_detector_skips_empty_values():
    """Empty data values are skipped — there's nothing to redact."""
    text = json.dumps({"kind": "Secret", "data": {"empty": "", "real": "dmFsdWU="}})

    matches = detect_kubernetes_secret(text)

    assert len(matches) == 1
    assert matches[0]["value"] == "dmFsdWU="


def test_detector_handles_repeated_values_without_overlap():
    """Same value in two slots produces two non-overlapping spans."""
    text = json.dumps({
        "kind": "SecretList",
        "items": [
            {"kind": "Secret", "data": {"k": "REPEATED"}},
            {"kind": "Secret", "data": {"k": "REPEATED"}},
        ],
    })

    matches = detect_kubernetes_secret(text)

    assert len(matches) == 2
    assert matches[0]["end"] <= matches[1]["start"]
    redacted = _redact(text)
    assert "REPEATED" not in redacted


# ============================================================================
# create_k8s_secret_pii_middleware
# ============================================================================


def test_factory_creates_pii_middleware_with_correct_flags():
    """Factory wires PIIMiddleware with the expected configuration."""
    mw = create_k8s_secret_pii_middleware()

    assert mw.pii_type == PII_TYPE
    assert mw.strategy == "redact"
    assert mw.apply_to_input is False
    assert mw.apply_to_output is False
    assert mw.apply_to_tool_results is True
