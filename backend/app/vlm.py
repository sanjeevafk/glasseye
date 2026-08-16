"""Provider-neutral advisory VLM review of a single evidence crop.

The VLM is strictly advisory: it reviews one selected YOLO evidence crop plus
structured metadata, and returns a constrained verdict (``confirm``,
``reject``, or ``escalate``) with a short rationale.  It never controls an
actuator; the policy engine remains the final decision-maker.

Modes
-----
- ``DEMO_VLM_MODE=fixture`` (default): deterministic rule-based reviewer for
  tests and presentation runs.  No network, no API key.
- ``DEMO_VLM_MODE=http``: OpenAI-compatible chat-completions provider,
  gated behind ``GLASSEYE_VLM_API_KEY``.  Any failure (missing credentials,
  timeout, non-200 response, malformed JSON, invalid verdict) raises
  :class:`VlmFailure`, which callers must map to REVIEW — never CLEAN.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Protocol

from .schemas import StrictModel, VlmReview, VlmVerdict


class VlmFailure(RuntimeError):
    """Raised when a VLM review cannot be produced safely."""


class VlmProvider(Protocol):
    def review(self, image_bytes: bytes, metadata: dict[str, Any]) -> VlmReview: ...


class FixtureVlmProvider:
    """Deterministic reviewer for tests and the repeatable presentation demo.

    The fixture is explicit about being a stand-in: it applies fixed rules to
    the structured metadata and never inspects the image pixels.
    """

    provider_name = "fixture-vlm"
    model_name = "deterministic-rules-v0"

    def review(self, image_bytes: bytes, metadata: dict[str, Any]) -> VlmReview:
        del image_bytes  # the fixture deliberately ignores pixels
        class_name = str(metadata.get("class_name", ""))
        confidence = float(metadata.get("confidence", 0.0))
        observations = int(metadata.get("observation_count", 0))
        started = time.perf_counter()
        if class_name == "structural_issue":
            verdict = VlmVerdict.ESCALATE
            rationale = "Structural defect present in the evidence crop; escalate for human structural review."
        elif confidence >= 0.65 and observations >= 3:
            verdict = VlmVerdict.CONFIRM
            rationale = "Visible cleanable surface issue consistent with the stable detection track."
        else:
            verdict = VlmVerdict.REJECT
            rationale = "Evidence crop does not clearly show a defect; recommend human review."
        return VlmReview(
            verdict=verdict,
            rationale=rationale,
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=max(1, int((time.perf_counter() - started) * 1000)),
        )


class _VlmResponse(StrictModel):
    verdict: VlmVerdict
    rationale: str


class HttpVlmProvider:
    """OpenAI-compatible vision-language review, gated behind an API key.

    Configuration (environment variables):
    - ``GLASSEYE_VLM_API_KEY`` (required)
    - ``GLASSEYE_VLM_BASE_URL`` (default ``https://api.openai.com/v1``)
    - ``GLASSEYE_VLM_MODEL`` (default ``gpt-4o-mini``)
    - ``GLASSEYE_VLM_TIMEOUT_SECONDS`` (default 10)
    """

    provider_name = "http-vlm"

    def __init__(self, transport: Any | None = None) -> None:
        self.api_key = os.environ.get("GLASSEYE_VLM_API_KEY", "")
        if not self.api_key:
            raise VlmFailure("GLASSEYE_VLM_API_KEY is not set; the real VLM mode is disabled.")
        self.base_url = os.environ.get("GLASSEYE_VLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("GLASSEYE_VLM_MODEL", "gpt-4o-mini")
        self.timeout_seconds = float(os.environ.get("GLASSEYE_VLM_TIMEOUT_SECONDS", "10"))
        self._transport = transport

    def review(self, image_bytes: bytes, metadata: dict[str, Any]) -> VlmReview:
        try:
            import httpx
        except ImportError as exc:
            raise VlmFailure("httpx is required for the HTTP VLM provider.") from exc
        prompt = (
            "You are an advisory facade-inspection reviewer. You receive ONE evidence "
            "crop of a possible building-façade defect plus structured detection metadata. "
            "Return strict JSON only: {\"verdict\": \"confirm\" | \"reject\" | \"escalate\", "
            "\"rationale\": \"one concise sentence\"}. "
            "confirm = the crop clearly shows the stated defect; "
            "reject = the crop does not show a defect (false detection); "
            "escalate = the defect looks structural or severe and needs human review. "
            "You never control an actuator; you only advise."
        )
        question = (
            f"Evidence metadata: class={metadata.get('class_name')}, "
            f"confidence={metadata.get('confidence')}, observations={metadata.get('observation_count')}, "
            f"panel={metadata.get('panel_id')}. "
            "Review this crop and return the strict JSON verdict."
        )
        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "temperature": 0,
                        "max_tokens": 200,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question},
                                    {"type": "image_url", "image_url": {"url": data_uri}},
                                ],
                            },
                        ],
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise VlmFailure(f"VLM request failed: {exc}") from exc
        try:
            parsed = _VlmResponse.model_validate_json(content)
        except Exception as exc:
            raise VlmFailure(f"VLM returned malformed output: {exc}") from exc
        return VlmReview(
            verdict=parsed.verdict,
            rationale=parsed.rationale,
            provider=self.provider_name,
            model=self.model,
            latency_ms=max(1, int((time.perf_counter() - started) * 1000)),
        )


def build_vlm_provider(mode: str | None = None) -> VlmProvider:
    """Select the provider from ``DEMO_VLM_MODE`` (default ``fixture``)."""
    selected = (mode or os.environ.get("DEMO_VLM_MODE", "fixture")).strip().lower()
    if selected == "fixture":
        return FixtureVlmProvider()
    if selected == "http":
        return HttpVlmProvider()
    raise ValueError(f"Unknown DEMO_VLM_MODE: {selected!r} (expected 'fixture' or 'http')")


def describe_metadata(
    *,
    issue_id: str,
    class_name: str,
    confidence: float,
    observation_count: int,
    bbox_xyxy: list[float],
    panel_id: str,
    model_version: str,
) -> dict[str, Any]:
    """Structured metadata submitted with the crop; never raw frames."""
    return {
        "issue_id": issue_id,
        "class_name": class_name,
        "confidence": round(confidence, 6),
        "observation_count": observation_count,
        "bbox_xyxy": [round(value, 3) for value in bbox_xyxy],
        "panel_id": panel_id,
        "model_version": model_version,
    }
