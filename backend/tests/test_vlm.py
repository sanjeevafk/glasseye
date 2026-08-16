from __future__ import annotations

import json

import httpx
import pytest

from app.schemas import VlmVerdict
from app.vlm import (
    FixtureVlmProvider,
    HttpVlmProvider,
    VlmFailure,
    build_vlm_provider,
)


def test_fixture_provider_returns_deterministic_verdicts() -> None:
    provider = FixtureVlmProvider()
    structural = provider.review(b"", {"class_name": "structural_issue", "confidence": 0.2, "observation_count": 1})
    cleanable_high = provider.review(
        b"", {"class_name": "cleanable_surface_issue", "confidence": 0.9, "observation_count": 5}
    )
    ambiguous = provider.review(
        b"", {"class_name": "cleanable_surface_issue", "confidence": 0.4, "observation_count": 2}
    )
    assert structural.verdict == VlmVerdict.ESCALATE
    assert cleanable_high.verdict == VlmVerdict.CONFIRM
    assert ambiguous.verdict == VlmVerdict.REJECT
    assert structural.provider == "fixture-vlm"
    assert structural.rationale


def test_build_provider_defaults_to_fixture() -> None:
    provider = build_vlm_provider()
    assert isinstance(provider, FixtureVlmProvider)


def test_build_provider_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        build_vlm_provider("telepathy")


def test_http_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLASSEYE_VLM_API_KEY", raising=False)
    with pytest.raises(VlmFailure):
        HttpVlmProvider()


def test_http_provider_parses_strict_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSEYE_VLM_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"verdict": "escalate", "rationale": "Crack is wide."}'}}
                ]
            },
        )

    provider = HttpVlmProvider(transport=httpx.MockTransport(handler))
    review = provider.review(b"jpeg-bytes", {"class_name": "structural_issue"})
    assert review.verdict == VlmVerdict.ESCALATE
    assert review.rationale == "Crack is wide."
    assert review.provider == "http-vlm"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"verdict": "maybe", "rationale": "x"}', "malformed"),
        ("not json at all", "malformed"),
        ('{"verdict": "confirm"}', "rationale"),
    ],
)
def test_http_provider_rejects_malformed_output(
    monkeypatch: pytest.MonkeyPatch, content: str, expected: str
) -> None:
    monkeypatch.setenv("GLASSEYE_VLM_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = HttpVlmProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(VlmFailure, match=expected):
        provider.review(b"jpeg-bytes", {})


def test_http_provider_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSEYE_VLM_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    provider = HttpVlmProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(VlmFailure):
        provider.review(b"jpeg-bytes", {})


def test_http_provider_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLASSEYE_VLM_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    provider = HttpVlmProvider(transport=httpx.MockTransport(handler))
    with pytest.raises(VlmFailure):
        provider.review(b"jpeg-bytes", {})
