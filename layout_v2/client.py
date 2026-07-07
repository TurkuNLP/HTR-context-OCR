"""Minimal OpenAI-compatible vLLM chat client with schema enforcement and targeted retries.

Re-implementation of the proven request pattern from the old pipeline (stdlib urllib — no extra
dependencies on the compute nodes), adapted for layout_v2's needs:
- multi-image user turns (full page + band crops);
- per-pass max_tokens / schema;
- always captures ``usage`` (prompt/completion tokens) — the input to the resolution audit.

The retry policy is deliberately narrow: only empty, unparseable, or schema-invalid model output
is retried; HTTP/transport errors are recorded and surfaced to the caller, which decides
per-document handling.
"""

from __future__ import annotations

import json
import os
import re
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import schema_check


# --------------------------------------------------------------------------------------
# Message construction helpers
# --------------------------------------------------------------------------------------
def text_part(text: str) -> dict:
    """One text segment of a multimodal chat message."""
    return {"type": "text", "text": text}


def image_part(data_url: str) -> dict:
    """One image segment of a multimodal chat message (base64 data URL)."""
    return {"type": "image_url", "image_url": {"url": data_url}}


def build_messages(system: str, user_parts: list[dict]) -> list[dict]:
    """Standard two-turn shape: system instructions + one user turn with text and image parts."""
    return [
        {"role": "system", "content": [text_part(system)]},
        {"role": "user", "content": user_parts},
    ]


# --------------------------------------------------------------------------------------
# Request / response
# --------------------------------------------------------------------------------------
def _flatten_content(content: Any) -> str:
    """Normalize chat message content (string or list-of-parts) into one text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content)


def _structured_output_fields(json_schema: dict) -> dict[str, Any]:
    """Return the request fields that ask vLLM to enforce our JSON schema.

    ``response_format`` is the default because it is the OpenAI-compatible structured-output path.
    ``LAYOUT_V2_STRUCTURED_OUTPUT=guided_json`` keeps the old vLLM-native mode available for
    quick A/B debugging on CSC module changes.
    """
    mode = os.getenv("LAYOUT_V2_STRUCTURED_OUTPUT", "response_format").strip().lower()
    if mode == "guided_json":
        return {"guided_json": json_schema}
    if mode in ("", "response_format"):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "layout_v2_pass",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        }
    raise ValueError(
        "LAYOUT_V2_STRUCTURED_OUTPUT must be 'response_format' or 'guided_json', "
        f"got {mode!r}"
    )


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    json_schema: dict | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    """POST one chat completion and return ``{content, reasoning_content, response_metadata}``.

    ``json_schema`` (when given) is requested server-side via structured outputs. The caller also
    validates client-side; the server is an accelerator for correctness, not our only guard.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if json_schema is not None:
        payload.update(_structured_output_fields(json_schema))

    request = Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    # Translate the two transport failure shapes into RuntimeErrors the caller records per-doc.
    try:
        with urlopen(request, timeout=max(timeout_seconds, 1)) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # server replied non-2xx (bad request, OOM, ...)
        details = exc.read().decode("utf-8", errors="ignore")[:2000]
        raise RuntimeError(f"vLLM HTTP {exc.code}: {details}") from exc
    except URLError as exc:  # could not reach the server at all
        raise RuntimeError(f"Failed to reach vLLM at {base_url}: {exc.reason}") from exc

    choices = parsed.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) or {}
    return {
        "content": _flatten_content(message.get("content")).strip(),
        # The Thinking variants' <think> trace, separated by the server's --reasoning-parser.
        "reasoning_content": _flatten_content(message.get("reasoning_content")).strip(),
        "response_metadata": {
            "model": parsed.get("model") or "",
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
            "usage": parsed.get("usage") or {},  # prompt/completion tokens -> resolution audit
            "structured_output_mode": os.getenv("LAYOUT_V2_STRUCTURED_OUTPUT", "response_format"),
        },
    }


# --------------------------------------------------------------------------------------
# Parsing + retry wrapper
# --------------------------------------------------------------------------------------
def parse_json_object(content: str) -> tuple[dict | None, str]:
    """Parse the model's answer into a dict; returns ``(obj_or_None, error_string)``.

    Structured decoding should hand us a clean object, but we defend against a Markdown fence and
    stray prose around the object (both observed in the wild with earlier vLLM builds).
    """
    text = content.strip()
    if not text:
        return None, "empty_content"
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)  # leading fence
    text = re.sub(r"\s*```\s*$", "", text)  # trailing fence
    try:
        value = json.loads(text)
        return (value, "") if isinstance(value, dict) else (None, "parsed_value_not_object")
    except json.JSONDecodeError as exc:
        first_error = f"json_decode_error: line {exc.lineno} col {exc.colno}: {exc.msg}"
    # Last resort: the first {...} span (handles prose wrapping).
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return (value, "") if isinstance(value, dict) else (None, "span_value_not_object")
        except json.JSONDecodeError as exc:
            return None, f"span_decode_error: line {exc.lineno} col {exc.colno}: {exc.msg}"
    return None, first_error


RETRYABLE_ERRORS = ("empty_content", "json_decode_error", "span_decode_error", "schema_validation_error")


def call_pass(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user_parts: list[dict],
    json_schema: dict,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> dict[str, Any]:
    """One pass = one structured-output request with retries; returns a self-describing result dict.

    Shape: ``{ok, parsed, parse_error, raw_content, reasoning_content, response_metadata,
    attempts, attempt_errors, error}``. Never raises — the runner records failures per document.
    """
    result: dict[str, Any] = {
        "ok": False,
        "parsed": None,
        "parse_error": "",
        "raw_content": "",
        "reasoning_content": "",
        "response_metadata": {},
        "attempts": 0,
        "attempt_errors": [],
        "error": "",
    }
    messages = build_messages(system, user_parts)
    for attempt in range(1, max_attempts + 1):
        result["attempts"] = attempt
        try:
            response = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                json_schema=json_schema,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # transport/server failure: record, do not blind-retry
            result["error"] = str(exc)
            result["attempt_errors"].append(f"attempt {attempt}: {exc}")
            break

        result["raw_content"] = response["content"]
        result["reasoning_content"] = response["reasoning_content"]
        result["response_metadata"] = response["response_metadata"]

        parsed, parse_error = parse_json_object(response["content"])
        if parsed is not None:
            schema_errors = schema_check.validate(parsed, json_schema)
            if schema_errors:
                parse_error = "schema_validation_error: " + schema_check.short_error(schema_errors)
                parsed = None
        result["parse_error"] = parse_error
        if parsed is not None:
            result["ok"] = True
            result["parsed"] = parsed
            result["error"] = ""
            break
        result["attempt_errors"].append(f"attempt {attempt}: {parse_error}")
        # Retry only output-contract failures; server/transport failures are handled above.
        if attempt < max_attempts and parse_error.startswith(RETRYABLE_ERRORS):
            sleep(backoff_seconds * attempt)
            continue
        break
    return result
