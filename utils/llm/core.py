"""Core chat/completion API for LLMs with provider fallback and caching."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

import litellm
from litellm import acompletion
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from churro.utils.log_utils import logger

from .config import DEFAULT_TIMEOUT, ensure_initialized
from .cost import cost_tracker
from .messages import prepare_messages
from .models import MODEL_MAP
from .types import ImageDetail, Messages, ModelInfo


class LLMInferenceError(RuntimeError):
    """Raised when all provider candidates fail or return unusable output."""


_EXACT_IO_ENV = "CHURRO_EXACT_IO_LOG"
_EXACT_IO_PATH_ENV = "CHURRO_EXACT_IO_LOG_PATH"
_EXACT_IO_PREFIX_ENV = "CHURRO_EXACT_IO_LOG_PREFIX"
_EXACT_IO_INCLUDE_RAW_RESPONSE_ENV = "CHURRO_EXACT_IO_INCLUDE_RAW_RESPONSE"
_EXACT_IO_DEFAULT_FILE = Path("logs") / "BENCHMARK_VLLM_EXACT_IO.jsonl"
_EXACT_IO_LOCK = threading.Lock()


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_exact_io_logging_enabled() -> bool:
    return _truthy(os.getenv(_EXACT_IO_ENV))


def _get_exact_io_log_path() -> Path:
    configured = os.getenv(_EXACT_IO_PATH_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    # core.py -> llm -> utils -> repo_root
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / _EXACT_IO_DEFAULT_FILE).resolve()


def _get_exact_io_log_prefix() -> Path:
    configured = os.getenv(_EXACT_IO_PREFIX_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    # Backward-compatible fallback: derive prefix from single-file path.
    single = _get_exact_io_log_path()
    return single.with_suffix("")


def _exact_io_file(event_type: str) -> Path:
    prefix = _get_exact_io_log_prefix()
    return Path(f"{prefix}_{event_type}.jsonl")


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_safe(value: Any) -> Any:
    """Convert nested objects to JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _serialize_response(response: Any) -> Any:
    """Best-effort full response serialization for audit logs."""
    try:
        if hasattr(response, "model_dump"):
            return _json_safe(response.model_dump())
        if hasattr(response, "dict"):
            return _json_safe(response.dict())
        if hasattr(response, "to_dict"):
            return _json_safe(response.to_dict())
    except Exception:  # pragma: no cover - defensive
        pass
    return _json_safe(response)


def _append_exact_io_record(event_type: str, record: dict[str, Any]) -> None:
    if not _is_exact_io_logging_enabled():
        return
    path = _exact_io_file(event_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_safe(record), ensure_ascii=False)
    with _EXACT_IO_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _short_text(value: Any, limit: int = 200) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


def _extract_data_url_stats(url: Any) -> dict[str, Any]:
    if not isinstance(url, str):
        return {"kind": "non-string-url"}
    if not url.startswith("data:"):
        return {
            "kind": "url",
            "prefix": _short_text(url, 160),
        }
    header, sep, data = url.partition(",")
    if not sep:
        return {
            "kind": "malformed-data-url",
            "header": _short_text(url, 120),
        }
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return {
        "kind": "data-url",
        "header": header,
        "base64_chars": len(data),
        "sha256_base64": digest,
    }


def _summarize_messages(messages: Any) -> Any:
    """Create compact message summary (no raw base64 image payload)."""
    if not isinstance(messages, list):
        return _json_safe(messages)

    summarized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            summarized.append({"raw": _json_safe(msg)})
            continue
        role = msg.get("role")
        content = msg.get("content")
        content_summary: list[dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    content_summary.append({"raw": _json_safe(item)})
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text_value = item.get("text")
                    content_summary.append(
                        {
                            "type": "text",
                            "chars": len(text_value) if isinstance(text_value, str) else None,
                            "preview": _short_text(text_value, 300),
                        }
                    )
                    continue
                if item_type == "image_url":
                    image_url_payload = item.get("image_url", {})
                    if not isinstance(image_url_payload, dict):
                        content_summary.append(
                            {
                                "type": "image_url",
                                "payload": _json_safe(image_url_payload),
                            }
                        )
                        continue
                    stats = _extract_data_url_stats(image_url_payload.get("url"))
                    content_summary.append(
                        {
                            "type": "image_url",
                            "detail": image_url_payload.get("detail"),
                            "url_stats": stats,
                        }
                    )
                    continue
                content_summary.append({"type": item_type, "payload": _json_safe(item)})
        else:
            content_summary.append({"raw_content": _json_safe(content)})

        summarized.append(
            {
                "role": role,
                "content_items": content_summary,
                "content_item_count": len(content_summary),
            }
        )
    return summarized


def _get_model_candidates(model_key: str) -> list[ModelInfo]:
    """Return ordered list of provider candidates for a logical model key."""
    candidates = MODEL_MAP.get(model_key)
    if candidates is None:
        raise ValueError(f"Unknown model key: {model_key}")
    return candidates


@retry(
    retry=retry_if_exception_type(
        (
            litellm.exceptions.APIError,
            litellm.exceptions.InternalServerError,
            litellm.exceptions.RateLimitError,
        )
    ),
    stop=stop_after_attempt(1),
    wait=wait_fixed(10),
)
async def _run_litellm(
    messages: Messages,
    model: str,
    output_json: bool = False,
    pydantic_class: type | None = None,  # For JSON schema validation
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run an LLM inference asynchronously."""
    # Lazy init of underlying litellm/caching setup
    ensure_initialized()
    candidates = _get_model_candidates(model)

    # Try each candidate in order until one yields a non-empty answer
    last_error: Exception | None = None
    empty_response_seen = False
    exact_io_enabled = _is_exact_io_logging_enabled()
    for candidate_index, candidate in enumerate(candidates):
        provider_model_original = candidate["provider_model"]
        provider_model = provider_model_original
        # Build params per-candidate
        additional_params: dict[str, Any] = {}
        if candidate.get("static_params"):
            additional_params.update(candidate["static_params"])  # type: ignore[index]
        if output_json:
            additional_params["response_format"] = {"type": "json_object"}
        if pydantic_class:
            additional_params["response_format"] = pydantic_class
        is_vllm_candidate = provider_model.startswith("vllm/")
        if is_vllm_candidate:
            # replace the model prefix for vllm, so that LiteLLM treats it as the OpenAI-compatible server that it is
            provider_model = provider_model.replace("vllm/", "openai/")
            num_retries = 1
        else:
            num_retries = 3

        try:
            request_started_at = _timestamp_utc()
            request_payload = {
                "model": provider_model,
                "messages": messages,
                "num_retries": num_retries,
                "timeout": timeout,
                "additional_params": additional_params,
            }
            if exact_io_enabled and is_vllm_candidate:
                common_fields = {
                    "timestamp_utc": request_started_at,
                    "model_key": model,
                    "candidate_index": candidate_index,
                    "provider_model_original": provider_model_original,
                    "provider_model_invoked": provider_model,
                    "request_started_at_utc": request_started_at,
                }
                _append_exact_io_record(
                    "summary",
                    {
                        "event": "request",
                        **common_fields,
                    }
                )
                _append_exact_io_record(
                    "request_meta",
                    {
                        "event": "request_meta",
                        **common_fields,
                        "timeout": timeout,
                        "num_retries": num_retries,
                        "additional_params": additional_params,
                    }
                )
                _append_exact_io_record(
                    "request_messages",
                    {
                        "event": "request_messages",
                        **common_fields,
                        "messages_summary": _summarize_messages(messages),
                    }
                )
            response = await acompletion(
                model=provider_model,
                messages=messages,
                num_retries=num_retries,
                timeout=timeout,
                **additional_params,
            )
            answer = response.choices[0].message.content  # type: ignore
            if exact_io_enabled and is_vllm_candidate:
                response_ts = _timestamp_utc()
                finish_reason = getattr(response.choices[0], "finish_reason", None)  # type: ignore[index]
                common_response_fields = {
                    "timestamp_utc": response_ts,
                    "model_key": model,
                    "candidate_index": candidate_index,
                    "provider_model_original": provider_model_original,
                    "provider_model_invoked": provider_model,
                    "request_started_at_utc": request_started_at,
                    "finish_reason": finish_reason,
                }
                _append_exact_io_record(
                    "summary",
                    {
                        "event": "response",
                        **common_response_fields,
                    }
                )
                _append_exact_io_record(
                    "response_meta",
                    {
                        "event": "response_meta",
                        **common_response_fields,
                        "usage": _json_safe(getattr(response, "usage", None)),
                        "hidden_params": _json_safe(getattr(response, "_hidden_params", None)),
                    }
                )
                _append_exact_io_record(
                    "response_text",
                    {
                        "event": "response_text",
                        **common_response_fields,
                        "answer_content": answer,
                    }
                )
                if _truthy(os.getenv(_EXACT_IO_INCLUDE_RAW_RESPONSE_ENV)):
                    _append_exact_io_record(
                        "response_raw",
                        {
                            "event": "response_raw",
                            **common_response_fields,
                            "response_payload": _serialize_response(response),
                        },
                    )
            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
                and response._hidden_params["response_cost"]
            ):
                cost_tracker.add_cost(response._hidden_params["response_cost"])

            if answer:
                return answer
            logger.error(
                f"LLM '{model}' via '{provider_model}' returned empty answer with finish_reason '{response.choices[0].finish_reason}'"  # type: ignore
            )
            empty_response_seen = True
        except Exception as e:  # Continue to next candidate on failure
            last_error = e
            if exact_io_enabled and is_vllm_candidate:
                error_ts = _timestamp_utc()
                common_error_fields = {
                    "timestamp_utc": error_ts,
                    "model_key": model,
                    "candidate_index": candidate_index,
                    "provider_model_original": provider_model_original,
                    "provider_model_invoked": provider_model,
                    "request_started_at_utc": request_started_at if "request_started_at" in locals() else None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                _append_exact_io_record(
                    "summary",
                    {
                        "event": "error",
                        **common_error_fields,
                    }
                )
                _append_exact_io_record(
                    "errors",
                    {
                        "event": "error",
                        **common_error_fields,
                    }
                )
            logger.warning(
                f"Provider '{provider_model}' failed for model key '{model}': {e}. Trying next candidate if available."
            )

    # All candidates failed or returned empty
    if last_error:
        message = (
            f"All provider candidates failed for model key '{model}'. Last error: {last_error}"
        )
        logger.error(message)
        raise LLMInferenceError(message) from last_error

    message = f"All provider candidates returned empty output for model key '{model}'."
    if empty_response_seen:
        logger.error(message)
    raise LLMInferenceError(message)


async def run_llm_async(
    model: str,
    system_prompt_text: str | None,
    user_message_text: str | None,
    user_message_image: Image.Image | list[Image.Image] | None = None,
    image_detail: ImageDetail | None = None,
    output_json: bool = False,
    pydantic_class: type | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Convenience wrapper around run_llm_async."""
    messages = prepare_messages(
        system_prompt_text,
        user_message_text,
        user_message_image,
        image_detail,
    )

    return await _run_litellm(
        messages,
        model,
        output_json=output_json,
        pydantic_class=pydantic_class,
        timeout=timeout,
    )
