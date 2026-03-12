#!/usr/bin/env python3
"""Standalone Churro OCR over the Hugging Face Churro dataset.

High-level flow:
1) Parse runtime arguments (backend, split, model/vLLM options, output paths).
2) Stream selected dataset split(s) from HF and filter to Finnish documents.
3) Run OCR generation using either:
   - local transformers model (`--backend transformers`), or
   - an external OpenAI-compatible vLLM endpoint (`--backend vllm`).
4) Extract XML, validate against schema, and derive clear text with tags removed.
5) Persist per-document artifacts as markdown:
   - model output markdown in `<output_dir>/<split>/model_results/<index>_<name>.md`
   - gold text markdown in `<output_dir>/<split>/gold/gold_<name>.md`
6) Write benchmark-style evaluation artifacts per split:
   - `<metrics_output_root>/<backend>/<split>/outputs.json`
   - `<metrics_output_root>/<backend>/<split>/all_metrics.json`

This script intentionally bypasses Churro CLI/Docker orchestration so it can be
run as a direct Python entry point for controlled experiments.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from time import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import get_dataset_split_names, load_dataset
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import xmlschema

# When executed from ./tests, add repository root so sibling modules are importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finnish_eval_copies import args as copied_args
from finnish_eval_copies.evaluation import metrics as copied_metrics
from finnish_eval_copies.evaluation import xml_utils as copied_xml_utils


CHURRO_DATASET_ID = copied_args.CHURRO_DATASET_ID
SCHEMA_PATH = copied_xml_utils.SCHEMA_PATH
extract_actual_text_from_xml = copied_xml_utils.extract_actual_text_from_xml
compute_metrics = copied_metrics.compute_metrics


# Runtime defaults for inference, output locations, and dataset behavior.
DEFAULT_MODEL_ID = "stanford-oval/churro-3B"
DEFAULT_SYSTEM_MESSAGE = "Transcribe the entiretly of this historical documents to XML format."
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "responses"
DATASET_SPLITS = ("dev", "test")
DEFAULT_DATASET_SPLIT = "all"
DEFAULT_BACKEND = "transformers"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_MODEL = "churro"
DEFAULT_METRICS_OUTPUT_ROOT: Path | None = None
MAX_IMAGE_DIM = 2500
MIN_PIXELS = 512 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
FINNISH_MAIN_LANGUAGE_FILTER = "finnish"
DEFAULT_TRANSCRIPTION_FIELD = "transcription"
DEV_TRANSCRIPTION_FIELD = "original_transcription"

_HISTORICAL_DOC_SCHEMA: xmlschema.XMLSchema | None = None


def _assert_using_copied_modules() -> None:
    """Ensure this test script uses local copied evaluation modules.

    The script must resolve imports from `finnish_eval_copies`, not from any
    globally installed package. This check fails early if module resolution is
    wrong, which prevents silently running with unexpected metric logic.
    """
    expected_root = (PROJECT_ROOT / "finnish_eval_copies").resolve()
    loaded_modules = {
        "copied_args": copied_args,
        "copied_metrics": copied_metrics,
        "copied_xml_utils": copied_xml_utils,
    }
    for module_name, module_obj in loaded_modules.items():
        module_file = getattr(module_obj, "__file__", None)
        if not module_file:
            raise RuntimeError(f"Module '{module_name}' has no __file__; cannot verify copy usage")
        resolved = Path(module_file).resolve()
        if expected_root not in resolved.parents:
            raise RuntimeError(
                f"Module '{module_name}' resolved outside copied package: {resolved}. "
                f"Expected location under: {expected_root}"
            )


def _parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments for inference and output control."""
    parser = argparse.ArgumentParser(
        description="Run standalone Churro inference over Hugging Face dataset images"
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["transformers", "vllm"],
        help=(
            "Inference backend. "
            "'transformers' uses local HF model loading. "
            "'vllm' calls an already-running OpenAI-compatible vLLM endpoint."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default=CHURRO_DATASET_ID,
        help=f"Hugging Face dataset id (default: {CHURRO_DATASET_ID})",
    )
    parser.add_argument(
        "--dataset-split",
        default=DEFAULT_DATASET_SPLIT,
        help=(
            "Dataset split to process (e.g. dev/test/train). "
            "Use 'all' to auto-discover and run every available split."
        ),
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Optional cap per split. 0 means process the full split.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="System prompt prepended to each dataset image.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20_000,
        help="Maximum number of tokens to generate per image.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature. 0 disables sampling.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' picks CUDA when available (transformers backend only).",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=DEFAULT_VLLM_BASE_URL,
        help=f"OpenAI-compatible vLLM base URL (default: {DEFAULT_VLLM_BASE_URL})",
    )
    parser.add_argument(
        "--vllm-model",
        default=DEFAULT_VLLM_MODEL,
        help=f"Served vLLM model name (default: {DEFAULT_VLLM_MODEL})",
    )
    parser.add_argument(
        "--vllm-api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for vLLM endpoint. Defaults to env OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument(
        "--vllm-timeout-seconds",
        type=int,
        default=600,
        help="HTTP timeout in seconds for each vLLM request.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help=(
            "Maximum concurrent in-flight requests for vLLM backend. "
            "For transformers backend this is ignored and execution stays sequential."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where markdown responses are saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--metrics-output-root",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT_ROOT,
        help=(
            "Root directory for benchmark-style evaluation outputs "
            "(outputs.json and all_metrics.json). Defaults to --output-dir."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose markdown output file already exists.",
    )
    return parser.parse_args()


def _iter_splits(*, dataset_id: str, split_arg: str) -> list[str]:
    """Resolve split selection so downstream code always iterates concrete split names."""
    normalized = split_arg.strip()
    if normalized and normalized != "all":
        return [normalized]
    if normalized != "all":
        raise ValueError("--dataset-split cannot be empty")

    try:
        raw_splits = get_dataset_split_names(dataset_id)
    except Exception as exc:
        fallback = list(DATASET_SPLITS)
        print(
            "[warn] "
            f"failed to auto-discover splits for dataset='{dataset_id}': {exc}. "
            f"Falling back to {fallback}."
        )
        return fallback

    resolved: list[str] = []
    seen: set[str] = set()
    for split_name in raw_splits:
        as_text = str(split_name).strip()
        if not as_text or as_text in seen:
            continue
        seen.add(as_text)
        resolved.append(as_text)

    if not resolved:
        raise RuntimeError(f"No splits discovered for dataset '{dataset_id}'")
    return resolved


def _inspect_split_fields(*, dataset_id: str, split: str) -> set[str]:
    """Inspect one split and return first-sample field names."""
    try:
        split_stream = load_dataset(dataset_id, split=split, streaming=True)
        for example in split_stream:
            if not isinstance(example, dict):
                raise RuntimeError(
                    f"expected dict sample while inspecting split='{split}', got {type(example)!r}"
                )
            return set(example.keys())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to inspect dataset fields for split='{split}' in dataset='{dataset_id}': {exc}"
        ) from exc
    return set()


def _transcription_field_for_split(split: str) -> str:
    """Return the transcription field expected for a given split.

    The CHURRO dev split exposes `original_transcription`, while other splits
    continue to use `transcription`.
    """
    return DEV_TRANSCRIPTION_FIELD if split.strip().casefold() == "dev" else DEFAULT_TRANSCRIPTION_FIELD


def _canonicalize_fields_for_comparison(fields: set[str]) -> set[str]:
    """Normalize split schema fields for cross-split compatibility checks."""
    canonical = set(fields)
    if DEFAULT_TRANSCRIPTION_FIELD in canonical or DEV_TRANSCRIPTION_FIELD in canonical:
        canonical.discard(DEFAULT_TRANSCRIPTION_FIELD)
        canonical.discard(DEV_TRANSCRIPTION_FIELD)
        canonical.add("__transcription_field__")
    return canonical


def _get_gold_transcription_for_split(*, example: dict[str, Any], split: str) -> str:
    """Read split-specific gold transcription text from a dataset example."""
    transcription_field = _transcription_field_for_split(split)
    return str(example.get(transcription_field) or "")


def _validate_split_fields(*, dataset_id: str, splits: list[str]) -> None:
    """Ensure selected splits expose consistent fields and required keys."""
    split_to_fields: dict[str, set[str]] = {}
    for split in splits:
        fields = _inspect_split_fields(dataset_id=dataset_id, split=split)
        split_to_fields[split] = fields
        if not fields:
            print(f"[warn] split='{split}' appears empty while checking field compatibility")

    non_empty = [(split, fields) for split, fields in split_to_fields.items() if fields]
    if not non_empty:
        print("[warn] field compatibility check saw no samples in selected splits")
        return

    reference_split, reference_fields = non_empty[0]
    canonical_reference_fields = _canonicalize_fields_for_comparison(reference_fields)
    mismatches: list[str] = []
    for split, fields in non_empty[1:]:
        canonical_fields = _canonicalize_fields_for_comparison(fields)
        if canonical_fields != canonical_reference_fields:
            missing = sorted(canonical_reference_fields - canonical_fields)
            extra = sorted(canonical_fields - canonical_reference_fields)
            mismatches.append(
                f"split='{split}' missing={missing or ['<none>']} extra={extra or ['<none>']}"
            )

    if mismatches:
        mismatch_summary = "; ".join(mismatches)
        raise RuntimeError(
            "Dataset split field mismatch detected. "
            f"Reference split='{reference_split}' fields={sorted(reference_fields)}. "
            f"Mismatches: {mismatch_summary}"
        )

    missing_required: list[str] = []
    required_field_set_descriptions: list[str] = []
    for split, fields in non_empty:
        required_fields = {"image", "main_language", _transcription_field_for_split(split)}
        required_field_set_descriptions.append(
            f"split='{split}':{sorted(required_fields)}"
        )
        missing = sorted(required_fields - fields)
        if missing:
            missing_required.append(f"split='{split}' missing={missing}")
    if missing_required:
        raise RuntimeError(
            "Selected split(s) are missing required fields for Finnish filtering/inference: "
            + "; ".join(missing_required)
        )

    print(
        "[schema] "
        f"checked splits={splits} shared_fields_count={len(reference_fields)} "
        f"required_fields_ok={'; '.join(required_field_set_descriptions)}"
    )


def _select_device(preference: str) -> torch.device:
    """Resolve device choice with validation for explicit CUDA requests."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    width, height = image.size
    if width <= max_width and height <= max_height:
        return image

    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if hasattr(Image, "Resampling"):
        resample_filter = Image.Resampling.LANCZOS
    else:  # pragma: no cover - Pillow < 10 fallback
        resample_filter = Image.LANCZOS  # type: ignore[attr-defined]
    return image.resize(new_size, resample=resample_filter)


def _dataset_image_to_pil(image_obj: Any) -> Image.Image:
    """Convert supported dataset image payloads into resized RGB PIL images.

    Supported payloads:
    - PIL.Image directly
    - dict payloads from HF image features (`bytes` or file `path`)
    - plain path-like strings
    """
    if isinstance(image_obj, Image.Image):
        return _resize_image_to_fit(image_obj.convert("RGB"), MAX_IMAGE_DIM, MAX_IMAGE_DIM)

    if isinstance(image_obj, dict):
        image_bytes = image_obj.get("bytes")
        image_path = image_obj.get("path")
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as source_image:
                image = source_image.convert("RGB")
            return _resize_image_to_fit(image, MAX_IMAGE_DIM, MAX_IMAGE_DIM)
        if image_path:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            return _resize_image_to_fit(image, MAX_IMAGE_DIM, MAX_IMAGE_DIM)

    if isinstance(image_obj, (str, Path)):
        with Image.open(image_obj) as source_image:
            image = source_image.convert("RGB")
        return _resize_image_to_fit(image, MAX_IMAGE_DIM, MAX_IMAGE_DIM)

    raise TypeError(f"Unsupported dataset image payload type: {type(image_obj)!r}")


def _load_processor(model_id: str) -> AutoProcessor:
    """Load model processor with explicit pixel bounds expected by Churro."""
    return AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )


def _prepare_inputs(
    *,
    processor: AutoProcessor,
    image: Image.Image,
    system_message: str,
    device: torch.device,
) -> dict[str, Any]:
    """Build model-ready tensors from image + prompt conversation.

    The processor's chat template converts system + user(image) messages into
    model-specific prompt text. Returned tensor inputs are moved to target device.
    """
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    )
    return {
        key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }


def _run_generation_transformers(
    *,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    inputs: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Run local transformer generation and decode only newly generated tokens."""
    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if processor.tokenizer.pad_token_id is not None:
        generation_kwargs.setdefault("pad_token_id", processor.tokenizer.pad_token_id)
    if processor.tokenizer.eos_token_id is not None:
        generation_kwargs.setdefault("eos_token_id", processor.tokenizer.eos_token_id)

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    new_tokens = generated[0, input_length:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _pil_to_data_url(image: Image.Image) -> str:
    """Encode PIL image as PNG data URL for OpenAI-compatible image input."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_vllm_response_text(payload: dict[str, Any]) -> str:
    """Extract text content from OpenAI chat completion response structure."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("vLLM response missing 'choices'")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                chunk_text = chunk.get("text")
                if isinstance(chunk_text, str) and chunk_text.strip():
                    text_parts.append(chunk_text.strip())
        if text_parts:
            return "\n".join(text_parts).strip()
    raise ValueError("vLLM response did not contain text content")


def _run_generation_vllm(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    image: Image.Image,
    system_message: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
    """Call vLLM HTTP endpoint and return model text response.

    This function sends a single multimodal chat request (system prompt + image),
    then normalizes common failure modes into readable RuntimeErrors.
    """
    data_url = _pil_to_data_url(image)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    request = Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(request, timeout=max(timeout_seconds, 1)) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        return _extract_vllm_response_text(parsed)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"vLLM HTTP error {exc.code} at {base_url.rstrip('/')}/chat/completions: {details}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to connect to vLLM at {base_url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("vLLM returned non-JSON response") from exc


def _run_generation_vllm_from_image_payload(
    *,
    image_payload: Any,
    base_url: str,
    api_key: str,
    model_name: str,
    system_message: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
    """Convenience wrapper for concurrent calls that receive raw dataset payloads."""
    image: Image.Image | None = None
    try:
        image = _dataset_image_to_pil(image_payload)
        return _run_generation_vllm(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            image=image,
            system_message=system_message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    finally:
        if image is not None:
            image.close()


def _extract_xml_payload(model_output: str) -> str:
    """Extract HistoricalDocument XML block from model output text.

    Handles fenced markdown code blocks and attempts to isolate the first full
    `<HistoricalDocument>...</HistoricalDocument>` segment when present.
    """
    text = model_output.strip()
    text = re.sub(r"^\s*```(?:xml)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(
        r"(<(?:\w+:)?HistoricalDocument\b.*?</(?:\w+:)?HistoricalDocument>)",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else text


def _extract_clear_text_from_xml_or_fallback(xml_or_text: str) -> str:
    """Extract clear text from XML, falling back to stripped raw text."""
    clear_text = extract_actual_text_from_xml(xml_or_text)
    if clear_text:
        return clear_text
    return xml_or_text.strip()


def _validate_xml_against_schema(xml_payload: str) -> bool:
    """Validate XML against XSD schema, lazily initializing shared schema object."""
    global _HISTORICAL_DOC_SCHEMA
    try:
        if _HISTORICAL_DOC_SCHEMA is None:
            _HISTORICAL_DOC_SCHEMA = xmlschema.XMLSchema(str(SCHEMA_PATH))
        return _HISTORICAL_DOC_SCHEMA.is_valid(xml_payload)
    except Exception:
        return False


def _slugify(value: str) -> str:
    """Convert arbitrary strings into filesystem-safe output name fragments."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "sample"

def _build_output_path(*, output_dir: Path, split: str, sample_index: int, file_name: str) -> Path:
    """Build markdown output path for one sample."""
    name_stem = Path(file_name).stem if file_name else f"sample_{sample_index:06d}"
    file_slug = _slugify(name_stem)
    model_results_dir = output_dir / split / "model_results"
    model_results_dir.mkdir(parents=True, exist_ok=True)
    return model_results_dir / f"{sample_index:06d}_{file_slug}.md"


def _build_gold_markdown_output_path(
    *,
    output_dir: Path,
    split: str,
    sample_index: int,
    file_name: str,
) -> Path:
    """Build per-document markdown path for extracted gold text."""
    name_stem = Path(file_name).stem if file_name else f"sample_{sample_index:06d}"
    file_slug = _slugify(name_stem)
    gold_dir = output_dir / split / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    return gold_dir / f"gold_{file_slug}.md"


def _build_metrics_output_prefix(*, metrics_output_root: Path, backend: str, split: str) -> Path:
    """Build benchmark artifact directory for one backend/split pair."""
    output_prefix = metrics_output_root / backend / split
    output_prefix.mkdir(parents=True, exist_ok=True)
    return output_prefix


def _to_eval_example(example: dict[str, Any], dataset_id: str, split: str) -> dict[str, Any]:
    """Keep only fields required by evaluation, with split-aware gold transcription."""
    document_type = str(example.get("document_type") or "print").strip().lower()
    if document_type not in {"print", "handwriting"}:
        document_type = "print"
    return {
        "file_name": str(example.get("file_name") or f"{split}_unknown"),
        "transcription": _get_gold_transcription_for_split(example=example, split=split),
        "main_language": str(example.get("main_language") or "unknown"),
        "main_script": str(example.get("main_script") or "unknown"),
        "document_type": document_type,
        "dataset_id": str(example.get("dataset_id") or dataset_id),
    }


def _format_main_language_and_type_levenshtein_metrics(
    metrics: dict[str, float | int],
) -> dict[str, dict[str, float | int]]:
    """Split flat language+type Levenshtein metrics into print/handwriting buckets."""
    print_metrics: dict[str, float | int] = {}
    handwriting_metrics: dict[str, float | int] = {}
    other_metrics: dict[str, float | int] = {}

    for key, value in metrics.items():
        if "_" not in key:
            other_metrics[key] = value
            continue
        language, kind = key.rsplit("_", 1)
        if kind == "print":
            print_metrics[language] = value
        elif kind == "handwriting":
            handwriting_metrics[language] = value
        else:
            other_metrics[key] = value

    formatted: dict[str, dict[str, float | int]] = {
        "print": {language: print_metrics[language] for language in sorted(print_metrics)},
        "handwriting": {
            language: handwriting_metrics[language] for language in sorted(handwriting_metrics)
        },
    }
    if other_metrics:
        formatted["other"] = {key: other_metrics[key] for key in sorted(other_metrics)}
    return formatted


def _should_include_example(*, example: dict[str, Any]) -> bool:
    """Keep only documents whose `main_language` is Finnish."""
    main_language = str(example.get("main_language") or "").strip().casefold()
    return main_language == FINNISH_MAIN_LANGUAGE_FILTER


def _write_markdown_response(
    *,
    output_path: Path,
    dataset_id: str,
    split: str,
    sample_index: int,
    file_name: str,
    model_id: str,
    system_message: str,
    raw_model_output: str,
    xml_payload: str,
    schema_valid: bool,
    plain_text: str,
) -> None:
    """Write human-readable markdown artifact containing prompt, XML, and clear text."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    markdown = (
        f"# Churro Dataset Inference Result\n\n"
        f"- Time (UTC): {timestamp}\n"
        f"- Dataset: `{dataset_id}`\n"
        f"- Split: `{split}`\n"
        f"- Sample Index: `{sample_index}`\n"
        f"- Source File: `{file_name}`\n"
        f"- Model: `{model_id}`\n\n"
        f"## System Prompt\n\n"
        f"{system_message}\n\n"
        f"## XML Transformation\n\n"
        f"- Schema file: `{SCHEMA_PATH}`\n"
        f"- XML schema valid: `{schema_valid}`\n\n"
        f"## Plain Text (XML Tags Removed)\n\n"
        f"{plain_text}\n\n"
        f"## Extracted XML Payload\n\n"
        f"```xml\n{xml_payload}\n```\n\n"
        f"## Raw Model Output\n\n"
        f"```xml\n{raw_model_output}\n```\n"
    )
    output_path.write_text(markdown, encoding="utf-8")


def _write_gold_markdown_response(
    *,
    output_path: Path,
    dataset_id: str,
    split: str,
    sample_index: int,
    file_name: str,
    model_id: str,
    golden_clear_text: str,
) -> None:
    """Write markdown artifact containing only the extracted gold/reference text."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    markdown = (
        f"# Churro Dataset Gold Text\n\n"
        f"- Time (UTC): {timestamp}\n"
        f"- Dataset: `{dataset_id}`\n"
        f"- Split: `{split}`\n"
        f"- Sample Index: `{sample_index}`\n"
        f"- Source File: `{file_name}`\n"
        f"- Model: `{model_id}`\n\n"
        f"## Golden Clear Text\n\n"
        f"{golden_clear_text}\n"
    )
    output_path.write_text(markdown, encoding="utf-8")


def _write_error_markdown(
    *,
    output_path: Path,
    dataset_id: str,
    split: str,
    sample_index: int,
    file_name: str,
    model_id: str,
    error_text: str,
) -> None:
    """Write markdown artifact for failures so errors are traceable per sample."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    markdown = (
        f"# Churro Dataset Inference Error\n\n"
        f"- Time (UTC): {timestamp}\n"
        f"- Dataset: `{dataset_id}`\n"
        f"- Split: `{split}`\n"
        f"- Sample Index: `{sample_index}`\n"
        f"- Source File: `{file_name}`\n"
        f"- Model: `{model_id}`\n\n"
        f"## Error\n\n"
        f"```text\n{error_text}\n```\n"
    )
    output_path.write_text(markdown, encoding="utf-8")


def main() -> int:
    """Main orchestration entry point.

    Handles model setup, dataset iteration, inference execution (sequential or
    concurrent), markdown artifact writing, and split-level metrics export.
    """
    _assert_using_copied_modules()
    args = _parse_args()
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")

    selected_splits = _iter_splits(dataset_id=args.dataset_id, split_arg=args.dataset_split)
    _validate_split_fields(dataset_id=args.dataset_id, splits=selected_splits)

    if args.backend == "transformers" and args.max_concurrency != 1:
        print(
            "[warn] --max-concurrency only applies to --backend vllm. "
            "Transformers backend runs sequentially."
        )

    device: torch.device | None = None
    processor: AutoProcessor | None = None
    model: AutoModelForImageTextToText | None = None

    # Only initialize local model weights for transformers backend.
    # The vLLM backend delegates generation to an external HTTP server.
    if args.backend == "transformers":
        device = _select_device(args.device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        processor = _load_processor(args.model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_root = args.metrics_output_root or args.output_dir
    metrics_output_root.mkdir(parents=True, exist_ok=True)

    # Global counters across all selected splits for final run summary.
    total_seen = 0
    total_success = 0
    total_failed = 0
    total_skipped = 0
    total_filtered_out_non_finnish = 0

    for split in selected_splits:
        split_eval_by_index: dict[int, dict[str, Any]] = {}
        split_predicted_by_index: dict[int, str] = {}
        split_start_time = time()
        split_filtered_out_non_finnish = 0

        print(f"[run] backend={args.backend} loading dataset split='{split}' from {args.dataset_id}")
        split_stream = load_dataset(args.dataset_id, split=split, streaming=True)

        # Count only examples that pass filtering and are considered for inference.
        processed_in_split = 0
        if args.backend == "vllm" and args.max_concurrency > 1:
            # Concurrent vLLM path:
            # submit futures until at capacity, then drain completed futures and
            # immediately persist artifacts + bookkeeping per finished sample.
            pending_futures: set[Future[str]] = set()
            future_meta: dict[Future[str], tuple[int, int, str, Path, str, str, dict[str, Any]]] = {}

            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                for sample_index, example in enumerate(split_stream):
                    # `max_samples_per_split` applies after filtering, so this cap
                    # counts only Finnish documents from the selected split.
                    if (
                        args.max_samples_per_split > 0
                        and processed_in_split >= args.max_samples_per_split
                    ):
                        break

                    if not isinstance(example, dict):
                        total_failed += 1
                        print(
                            f"[error] split={split} idx={sample_index} unexpected sample type: "
                            f"{type(example)!r}"
                        )
                        continue

                    # Keep only Finnish samples (`main_language == Finnish`).
                    if not _should_include_example(example=example):
                        split_filtered_out_non_finnish += 1
                        total_filtered_out_non_finnish += 1
                        continue

                    processed_in_split += 1
                    total_seen += 1
                    output_file_index = processed_in_split

                    file_name = str(example.get("file_name") or f"{split}_{sample_index:06d}")
                    output_path = _build_output_path(
                        output_dir=args.output_dir,
                        split=split,
                        sample_index=output_file_index,
                        file_name=file_name,
                    )

                    # Optional resume mode: skip work if markdown output exists.
                    if args.skip_existing and output_path.exists():
                        total_skipped += 1
                        print(f"[skip] split={split} idx={sample_index} output exists -> {output_path}")
                        continue

                    eval_item = _to_eval_example(example, args.dataset_id, split)
                    gold_transcription = _get_gold_transcription_for_split(
                        example=example,
                        split=split,
                    )
                    # Submit request to worker pool and remember metadata needed
                    # when this future completes.
                    future = executor.submit(
                        _run_generation_vllm_from_image_payload,
                        image_payload=example.get("image"),
                        base_url=args.vllm_base_url,
                        api_key=args.vllm_api_key,
                        model_name=args.vllm_model,
                        system_message=args.system_message,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        timeout_seconds=args.vllm_timeout_seconds,
                    )
                    pending_futures.add(future)
                    future_meta[future] = (
                        sample_index,
                        output_file_index,
                        file_name,
                        output_path,
                        args.vllm_model,
                        gold_transcription,
                        eval_item,
                    )

                    # Backpressure: when at capacity, wait for at least one future
                    # to complete before submitting more.
                    while len(pending_futures) >= args.max_concurrency:
                        done_futures, pending_futures = wait(
                            pending_futures,
                            return_when=FIRST_COMPLETED,
                        )
                        for done_future in done_futures:
                            (
                                sample_idx,
                                sample_output_file_idx,
                                sample_file_name,
                                sample_output_path,
                                model_label,
                                gold_transcription,
                                eval_item,
                            ) = (
                                future_meta.pop(done_future)
                            )
                            try:
                                raw_output = done_future.result()
                                xml_payload = _extract_xml_payload(raw_output)
                                schema_valid = _validate_xml_against_schema(xml_payload)
                                plain_text = _extract_clear_text_from_xml_or_fallback(xml_payload)
                                gold_text = _extract_clear_text_from_xml_or_fallback(gold_transcription)
                                gold_output_path = _build_gold_markdown_output_path(
                                    output_dir=args.output_dir,
                                    split=split,
                                    sample_index=sample_output_file_idx,
                                    file_name=sample_file_name,
                                )
                                _write_markdown_response(
                                    output_path=sample_output_path,
                                    dataset_id=args.dataset_id,
                                    split=split,
                                    sample_index=sample_idx,
                                    file_name=sample_file_name,
                                    model_id=model_label,
                                    system_message=args.system_message,
                                    raw_model_output=raw_output,
                                    xml_payload=xml_payload,
                                    schema_valid=schema_valid,
                                    plain_text=plain_text,
                                )
                                _write_gold_markdown_response(
                                    output_path=gold_output_path,
                                    dataset_id=args.dataset_id,
                                    split=split,
                                    sample_index=sample_idx,
                                    file_name=sample_file_name,
                                    model_id=model_label,
                                    golden_clear_text=gold_text,
                                )
                                total_success += 1
                                split_eval_by_index[sample_idx] = eval_item
                                split_predicted_by_index[sample_idx] = raw_output
                                print(
                                    f"[ok] split={split} idx={sample_idx} "
                                    f"-> {sample_output_path} | {gold_output_path}"
                                )
                            except Exception as exc:
                                _write_error_markdown(
                                    output_path=sample_output_path,
                                    dataset_id=args.dataset_id,
                                    split=split,
                                    sample_index=sample_idx,
                                    file_name=sample_file_name,
                                    model_id=model_label,
                                    error_text=str(exc),
                                )
                                total_failed += 1
                                split_eval_by_index[sample_idx] = eval_item
                                split_predicted_by_index[sample_idx] = ""
                                print(f"[error] split={split} idx={sample_idx}: {exc}")

                # Drain any remaining in-flight futures after stream iteration ends.
                while pending_futures:
                    done_futures, pending_futures = wait(
                        pending_futures,
                        return_when=FIRST_COMPLETED,
                    )
                    for done_future in done_futures:
                        (
                            sample_idx,
                            sample_output_file_idx,
                            sample_file_name,
                            sample_output_path,
                            model_label,
                            gold_transcription,
                            eval_item,
                        ) = (
                            future_meta.pop(done_future)
                        )
                        try:
                            raw_output = done_future.result()
                            xml_payload = _extract_xml_payload(raw_output)
                            schema_valid = _validate_xml_against_schema(xml_payload)
                            plain_text = _extract_clear_text_from_xml_or_fallback(xml_payload)
                            gold_text = _extract_clear_text_from_xml_or_fallback(gold_transcription)
                            gold_output_path = _build_gold_markdown_output_path(
                                output_dir=args.output_dir,
                                split=split,
                                sample_index=sample_output_file_idx,
                                file_name=sample_file_name,
                            )
                            _write_markdown_response(
                                output_path=sample_output_path,
                                dataset_id=args.dataset_id,
                                split=split,
                                sample_index=sample_idx,
                                file_name=sample_file_name,
                                model_id=model_label,
                                system_message=args.system_message,
                                raw_model_output=raw_output,
                                xml_payload=xml_payload,
                                schema_valid=schema_valid,
                                plain_text=plain_text,
                            )
                            _write_gold_markdown_response(
                                output_path=gold_output_path,
                                dataset_id=args.dataset_id,
                                split=split,
                                sample_index=sample_idx,
                                file_name=sample_file_name,
                                model_id=model_label,
                                golden_clear_text=gold_text,
                            )
                            total_success += 1
                            split_eval_by_index[sample_idx] = eval_item
                            split_predicted_by_index[sample_idx] = raw_output
                            print(
                                f"[ok] split={split} idx={sample_idx} "
                                f"-> {sample_output_path} | {gold_output_path}"
                            )
                        except Exception as exc:
                            _write_error_markdown(
                                output_path=sample_output_path,
                                dataset_id=args.dataset_id,
                                split=split,
                                sample_index=sample_idx,
                                file_name=sample_file_name,
                                model_id=model_label,
                                error_text=str(exc),
                            )
                            total_failed += 1
                            split_eval_by_index[sample_idx] = eval_item
                            split_predicted_by_index[sample_idx] = ""
                            print(f"[error] split={split} idx={sample_idx}: {exc}")
        else:
            # Sequential path (transformers backend, or vLLM with concurrency=1).
            for sample_index, example in enumerate(split_stream):
                if args.max_samples_per_split > 0 and processed_in_split >= args.max_samples_per_split:
                    break

                if not isinstance(example, dict):
                    total_failed += 1
                    print(f"[error] split={split} idx={sample_index} unexpected sample type: {type(example)!r}")
                    continue

                if not _should_include_example(example=example):
                    split_filtered_out_non_finnish += 1
                    total_filtered_out_non_finnish += 1
                    continue

                processed_in_split += 1
                total_seen += 1
                output_file_index = processed_in_split

                file_name = str(example.get("file_name") or f"{split}_{sample_index:06d}")
                output_path = _build_output_path(
                    output_dir=args.output_dir,
                    split=split,
                    sample_index=output_file_index,
                    file_name=file_name,
                )

                if args.skip_existing and output_path.exists():
                    total_skipped += 1
                    print(f"[skip] split={split} idx={sample_index} output exists -> {output_path}")
                    continue

                eval_item = _to_eval_example(example, args.dataset_id, split)
                active_model_label = args.vllm_model if args.backend == "vllm" else args.model_id
                image: Image.Image | None = None

                try:
                    image = _dataset_image_to_pil(example["image"])

                    # Choose inference execution path based on selected backend.
                    if args.backend == "vllm":
                        raw_output = _run_generation_vllm(
                            base_url=args.vllm_base_url,
                            api_key=args.vllm_api_key,
                            model_name=args.vllm_model,
                            image=image,
                            system_message=args.system_message,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            timeout_seconds=args.vllm_timeout_seconds,
                        )
                    else:
                        assert processor is not None
                        assert model is not None
                        assert device is not None
                        inputs = _prepare_inputs(
                            processor=processor,
                            image=image,
                            system_message=args.system_message,
                            device=device,
                        )
                        raw_output = _run_generation_transformers(
                            model=model,
                            processor=processor,
                            inputs=inputs,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )

                    xml_payload = _extract_xml_payload(raw_output)
                    schema_valid = _validate_xml_against_schema(xml_payload)
                    plain_text = _extract_clear_text_from_xml_or_fallback(xml_payload)
                    gold_text = _extract_clear_text_from_xml_or_fallback(
                        _get_gold_transcription_for_split(example=example, split=split)
                    )
                    gold_output_path = _build_gold_markdown_output_path(
                        output_dir=args.output_dir,
                        split=split,
                        sample_index=output_file_index,
                        file_name=file_name,
                    )

                    _write_markdown_response(
                        output_path=output_path,
                        dataset_id=args.dataset_id,
                        split=split,
                        sample_index=sample_index,
                        file_name=file_name,
                        model_id=active_model_label,
                        system_message=args.system_message,
                        raw_model_output=raw_output,
                        xml_payload=xml_payload,
                        schema_valid=schema_valid,
                        plain_text=plain_text,
                    )
                    _write_gold_markdown_response(
                        output_path=gold_output_path,
                        dataset_id=args.dataset_id,
                        split=split,
                        sample_index=sample_index,
                        file_name=file_name,
                        model_id=active_model_label,
                        golden_clear_text=gold_text,
                    )

                    total_success += 1
                    split_eval_by_index[sample_index] = eval_item
                    split_predicted_by_index[sample_index] = raw_output
                    print(
                        f"[ok] split={split} idx={sample_index} "
                        f"-> {output_path} | {gold_output_path}"
                    )
                except Exception as exc:
                    total_failed += 1
                    split_eval_by_index[sample_index] = eval_item
                    split_predicted_by_index[sample_index] = ""
                    _write_error_markdown(
                        output_path=output_path,
                        dataset_id=args.dataset_id,
                        split=split,
                        sample_index=sample_index,
                        file_name=file_name,
                        model_id=active_model_label,
                        error_text=str(exc),
                    )
                    print(f"[error] split={split} idx={sample_index}: {exc}")
                finally:
                    if image is not None:
                        image.close()

        print(
            "[filter] "
            f"split={split} main_language='Finnish' kept={processed_in_split} "
            f"filtered_non_finnish={split_filtered_out_non_finnish}"
        )

        split_elapsed = time() - split_start_time
        split_order = sorted(split_eval_by_index.keys())
        split_eval_dataset = [split_eval_by_index[idx] for idx in split_order]
        split_predicted_texts = [split_predicted_by_index[idx] for idx in split_order]

        if split_eval_dataset:
            metrics_output_prefix = _build_metrics_output_prefix(
                metrics_output_root=metrics_output_root,
                backend=args.backend,
                split=split,
            )
            combined_metrics = compute_metrics(
                split_eval_dataset,
                split_predicted_texts,
                str(metrics_output_prefix),
                elapsed_time=split_elapsed,
            )
            sorted_lang_type = _format_main_language_and_type_levenshtein_metrics(
                combined_metrics.get("main_language_and_type_metrics", {})
            )
            combined_metrics["main_language_and_type_metrics"] = sorted_lang_type

            all_metrics_path = metrics_output_prefix / "all_metrics.json"
            with open(all_metrics_path, "w", encoding="utf-8") as f:
                json.dump(combined_metrics, f, indent=2)
            print(f"[metrics] split={split} -> {all_metrics_path}")
        else:
            print(f"[metrics] split={split} skipped (no evaluated samples)")

        print(f"[split] split={split} elapsed_seconds={split_elapsed:.2f}")

    print(
        "[done] "
        f"seen={total_seen} success={total_success} failed={total_failed} skipped={total_skipped} "
        f"filtered_non_finnish_total={total_filtered_out_non_finnish} "
        f"output_dir={args.output_dir} metrics_output_root={metrics_output_root}"
    )
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
