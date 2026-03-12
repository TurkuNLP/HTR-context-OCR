#!/usr/bin/env python3
"""Minimal one-image finetuned OCR call against a locally served vLLM endpoint.

This script uses the same repository path as `--system finetuned`:
- engine key (default: `churro`) from `MODEL_MAP`
- system prompt used by `FineTunedOCR`
- local OpenAI-compatible vLLM endpoint (`http://localhost:<port>/v1`)
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
from openai import OpenAI


DEFAULT_IMAGE = Path("tests/churro_dataset_sample_1.jpeg")
DEFAULT_ENGINE = "churro"
DEFAULT_PROMPT = "Show me the input you are expecting. Print out the possible parameters you need. How fo I actvate the finetuned version of yourself? Don't give me the output file, show me exact input and processing."
DEFAULT_OUTPUT_DIR = Path("responses")
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_LOCAL_VLLM_PORT = 8000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one finetuned OCR request against local vLLM using Churro internals."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image path (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help=f"Logical engine key from MODEL_MAP (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--system-message",
        default=DEFAULT_PROMPT,
        help="System prompt sent with the image.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"OpenAI client timeout for the vLLM request (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--local-vllm-port",
        type=int,
        default=DEFAULT_LOCAL_VLLM_PORT,
        help=f"Local vLLM port (default: {DEFAULT_LOCAL_VLLM_PORT}).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="OPENAI_API_KEY value for local vLLM auth (default: env OPENAI_API_KEY or EMPTY).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for markdown output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit markdown output file path.",
    )
    parser.add_argument(
        "--strip-xml",
        action="store_true",
        help="Also extract plain text via evaluation.xml_utils.extract_actual_text_from_xml.",
    )
    return parser.parse_args()


def _configure_runtime_env(local_vllm_port: int, openai_api_key: str) -> None:
    os.environ["USE_EXISTING_VLLM"] = "1"
    os.environ["LOCAL_VLLM_PORT"] = str(local_vllm_port)
    os.environ["OPENAI_API_KEY"] = openai_api_key


def _build_output_path(output_dir: Path, explicit_file: Path | None) -> Path:
    if explicit_file is not None:
        explicit_file.parent.mkdir(parents=True, exist_ok=True)
        return explicit_file
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"single_finetuned_vllm_{ts}.md"


def _model_name_for_vllm(provider_model: str, fallback_engine: str) -> str:
    provider_model = provider_model.strip()
    for prefix in ("vllm/", "openai/"):
        if provider_model.startswith(prefix):
            model_name = provider_model[len(prefix) :].strip()
            if model_name:
                return model_name
    return fallback_engine


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts).strip()
    return ""


def _pil_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _run_once(args: argparse.Namespace) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    _configure_runtime_env(args.local_vllm_port, args.openai_api_key)

    from churro.config.settings import get_settings as get_churro_settings
    from churro.evaluation.xml_utils import extract_actual_text_from_xml
    from churro.utils.llm.config import get_settings as get_llm_settings
    from churro.utils.llm.models import MODEL_MAP, reload_model_map

    get_churro_settings(reload=True)
    get_llm_settings.cache_clear()
    reload_model_map()

    if args.engine not in MODEL_MAP:
        raise ValueError(f"Unknown engine key '{args.engine}'. Available keys: {sorted(MODEL_MAP)}")

    model_cfg = MODEL_MAP[args.engine][0]
    static_params = model_cfg.get("static_params", {}) if isinstance(model_cfg, dict) else {}
    if not isinstance(static_params, dict):
        static_params = {}

    provider_model = str(model_cfg.get("provider_model") or args.engine)
    if not provider_model.startswith(("vllm/", "openai/")):
        raise ValueError(
            f"Engine '{args.engine}' is mapped to provider_model '{provider_model}', "
            "not a local vLLM-compatible entry. Use --engine churro for this script."
        )
    request_model = _model_name_for_vllm(provider_model, args.engine)
    api_base = str(
        static_params.get("api_base")
        or f"http://localhost:{args.local_vllm_port}/v1"
    ).rstrip("/")
    temperature = float(static_params.get("temperature", 0.6))
    max_tokens = int(model_cfg.get("max_completion_tokens") or 20000)

    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be >= 1")

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    with Image.open(args.image) as source:
        image = source.convert("RGB")
        image_data_url = _pil_to_data_url(image)

    client = OpenAI(
        api_key=args.openai_api_key,
        base_url=api_base,
        timeout=max(args.timeout_seconds, 1),
    )
    response = client.chat.completions.create(
        model=request_model,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": args.system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_data_url}}],
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not response.choices:
        raise RuntimeError("vLLM returned no choices.")
    raw_output = _extract_response_text(response.choices[0].message.content)
    if not raw_output:
        raise RuntimeError("vLLM returned an empty response text.")

    request_config = {
        "provider_model": provider_model,
        "request_model": request_model,
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "local_vllm_port": args.local_vllm_port,
        "use_existing_vllm": os.getenv("USE_EXISTING_VLLM"),
    }

    plain_text = ""
    if args.strip_xml:
        try:
            plain_text = extract_actual_text_from_xml(raw_output) or ""
        except Exception:
            plain_text = ""

    return raw_output, plain_text, model_cfg, request_config


def _write_markdown(
    *,
    output_path: Path,
    args: argparse.Namespace,
    raw_output: str,
    plain_text: str,
    model_cfg: dict[str, Any],
    request_config: dict[str, Any],
) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base_url = f"http://localhost:{args.local_vllm_port}/v1"
    markdown = (
        "# Single Finetuned OCR Request\n\n"
        f"- Time (UTC): `{ts}`\n"
        f"- Image: `{args.image}`\n"
        f"- Engine key: `{args.engine}`\n"
        f"- Local vLLM base URL: `{base_url}`\n"
        f"- OPENAI_API_KEY set: `{'yes' if bool(args.openai_api_key) else 'no'}`\n\n"
        "## Repository Model Flags (from MODEL_MAP)\n\n"
        "```json\n"
        f"{json.dumps(model_cfg, indent=2, ensure_ascii=False)}\n"
        "```\n\n"
        "## Effective vLLM Request Config\n\n"
        "```json\n"
        f"{json.dumps(request_config, indent=2, ensure_ascii=False)}\n"
        "```\n\n"
        "## System Prompt\n\n"
        f"{args.system_message}\n\n"
        "## Raw Model Output\n\n"
        "```xml\n"
        f"{raw_output}\n"
        "```\n"
    )
    if args.strip_xml:
        markdown += (
            "\n## Extracted Plain Text\n\n"
            "```text\n"
            f"{plain_text}\n"
            "```\n"
        )
    output_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_path = _build_output_path(args.output_dir, args.output_file)
    try:
        raw_output, plain_text, model_cfg, request_config = _run_once(args)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    _write_markdown(
        output_path=output_path,
        args=args,
        raw_output=raw_output,
        plain_text=plain_text,
        model_cfg=model_cfg,
        request_config=request_config,
    )

    print(f"[ok] Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
