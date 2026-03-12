#!/usr/bin/env python3
"""Run Churro OCR on local churchbook images without evaluation metrics.

For each input image, this script writes two markdown files under
`results/churchbook_results`:
- `xml_results/<image_stem>_xml.md`: extracted XML payload
- `_pure_text_results/<image_stem>_pure_text.md`: plain text extracted from that XML payload
"""

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
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.xml_utils import extract_actual_text_from_xml


DEFAULT_INPUT_DIR = Path("/scratch/project_2017385/dorian/Churro_churchbooks/churchbook_images")
DEFAULT_OUTPUT_ROOT = Path("/scratch/project_2017385/dorian/Churro_churchbooks/results/churchbook_results")
DEFAULT_MODEL_ID = "stanford-oval/churro-3B"
DEFAULT_SYSTEM_MESSAGE = "Transcribe the entiretly of this historical documents to XML format."
DEFAULT_BACKEND = "vllm"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_MODEL = "churro"
DEFAULT_MAX_CONCURRENCY = 1
MAX_IMAGE_DIM = 2500
MIN_PIXELS = 512 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run churchbook image inference with Churro and save XML + pure text markdown outputs."
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["transformers", "vllm"],
        help=(
            "Inference backend. "
            "'transformers' loads a local HF model. "
            "'vllm' uses an already-running OpenAI-compatible endpoint."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing churchbook images (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for markdown outputs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id for transformers backend (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--system-message",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="System prompt prepended to each image request.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20_000,
        help="Maximum number of generated tokens per image.",
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
        help="Computation device for transformers backend only.",
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
        help="API key for vLLM endpoint. Defaults to OPENAI_API_KEY or EMPTY.",
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
        default=DEFAULT_MAX_CONCURRENCY,
        help="Maximum concurrent in-flight requests for vLLM backend.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on number of images to process. 0 means all images.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip image when both output markdown files already exist.",
    )
    return parser.parse_args()


def _select_device(preference: str) -> torch.device:
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

    scale = min(max_width / float(width), max_height / float(height))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if hasattr(Image, "Resampling"):
        resample_filter = Image.Resampling.LANCZOS
    else:  # pragma: no cover
        resample_filter = Image.LANCZOS  # type: ignore[attr-defined]
    return image.resize(new_size, resample=resample_filter)


def _load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    return _resize_image_to_fit(image, MAX_IMAGE_DIM, MAX_IMAGE_DIM)


def _load_processor(model_id: str) -> AutoProcessor:
    return AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )


def _prepare_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    system_message: str,
    device: torch.device,
) -> Dict[str, Any]:
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
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    inputs: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
) -> str:
    input_ids = inputs["input_ids"]
    input_length = input_ids.shape[1]

    generation_kwargs: Dict[str, Any] = {
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
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64,{}".format(encoded)


def _extract_vllm_response_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("vLLM response missing 'choices'")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                chunk_text = chunk.get("text")
                if isinstance(chunk_text, str) and chunk_text.strip():
                    text_parts.append(chunk_text.strip())
        if text_parts:
            return "\n".join(text_parts).strip()
    raise ValueError("vLLM response did not contain text content")


def _run_generation_vllm(
    base_url: str,
    api_key: str,
    model_name: str,
    image: Image.Image,
    system_message: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
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
        url="{}/chat/completions".format(base_url.rstrip("/")),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(api_key),
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
            "vLLM HTTP error {} at {}/chat/completions: {}".format(
                exc.code, base_url.rstrip("/"), details
            )
        )
    except URLError as exc:
        raise RuntimeError("Failed to connect to vLLM at {}: {}".format(base_url, exc.reason))
    except json.JSONDecodeError:
        raise RuntimeError("vLLM returned non-JSON response")


def _extract_xml_payload(model_output: str) -> str:
    text = model_output.strip()
    text = re.sub(r"^\s*```(?:xml)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(
        r"(<(?:\w+:)?HistoricalDocument\b.*?</(?:\w+:)?HistoricalDocument>)",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else text


def _iter_input_images(input_dir: Path) -> List[Path]:
    files = []
    for candidate in input_dir.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            files.append(candidate)
    return sorted(files)


def _build_output_paths(output_root: Path, input_root: Path, image_path: Path) -> Tuple[Path, Path]:
    relative_parent = image_path.parent.relative_to(input_root)
    xml_target_dir = output_root / "xml_results" / relative_parent
    text_target_dir = output_root / "_pure_text_results" / relative_parent
    xml_target_dir.mkdir(parents=True, exist_ok=True)
    text_target_dir.mkdir(parents=True, exist_ok=True)
    image_stem = image_path.stem
    xml_path = xml_target_dir / "{}_xml.md".format(image_stem)
    text_path = text_target_dir / "{}_pure_text.md".format(image_stem)
    return xml_path, text_path


def _write_xml_markdown(
    output_path: Path,
    image_path: Path,
    model_label: str,
    system_message: str,
    xml_payload: str,
    raw_model_output: str,
) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    markdown = (
        "# Churchbook XML Output\n\n"
        "- Time (UTC): `{}`\n"
        "- Image: `{}`\n"
        "- Model: `{}`\n\n"
        "## System Prompt\n\n"
        "{}\n\n"
        "## Extracted XML Payload\n\n"
        "```xml\n{}\n```\n\n"
        "## Raw Model Output\n\n"
        "```xml\n{}\n```\n".format(
            timestamp,
            image_path.name,
            model_label,
            system_message,
            xml_payload,
            raw_model_output,
        )
    )
    output_path.write_text(markdown, encoding="utf-8")


def _write_plain_text_markdown(
    output_path: Path,
    image_path: Path,
    model_label: str,
    plain_text: str,
) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    markdown = (
        "# Churchbook Plain Text Output\n\n"
        "- Time (UTC): `{}`\n"
        "- Image: `{}`\n"
        "- Model: `{}`\n\n"
        "## Plain Text\n\n"
        "{}\n".format(
            timestamp,
            image_path.name,
            model_label,
            plain_text,
        )
    )
    output_path.write_text(markdown, encoding="utf-8")


def _infer_vllm_from_image_path(
    image_path: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    system_message: str,
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
    image = None
    try:
        image = _load_image(image_path)
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


def _save_outputs_for_image(
    image_path: Path,
    xml_output_path: Path,
    text_output_path: Path,
    model_label: str,
    system_message: str,
    raw_output: str,
) -> None:
    xml_payload = _extract_xml_payload(raw_output)
    plain_text = extract_actual_text_from_xml(xml_payload)
    if not plain_text:
        plain_text = raw_output.strip()
    _write_xml_markdown(
        output_path=xml_output_path,
        image_path=image_path,
        model_label=model_label,
        system_message=system_message,
        xml_payload=xml_payload,
        raw_model_output=raw_output,
    )
    _write_plain_text_markdown(
        output_path=text_output_path,
        image_path=image_path,
        model_label=model_label,
        plain_text=plain_text,
    )


def _run_vllm_with_concurrency(args: argparse.Namespace, image_paths: List[Path]) -> Tuple[int, int, int, int]:
    seen = 0
    success = 0
    failed = 0
    skipped = 0
    pending_futures = set()  # type: Set[Future]
    future_meta = {}  # type: Dict[Future, Tuple[Path, Path, Path]]

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        for image_path in image_paths:
            xml_path, text_path = _build_output_paths(args.output_root, args.input_dir, image_path)
            seen += 1

            if args.skip_existing and xml_path.exists() and text_path.exists():
                skipped += 1
                print("[skip] image={} outputs exist".format(image_path))
                continue

            future = executor.submit(
                _infer_vllm_from_image_path,
                image_path,
                args.vllm_base_url,
                args.vllm_api_key,
                args.vllm_model,
                args.system_message,
                args.max_new_tokens,
                args.temperature,
                args.vllm_timeout_seconds,
            )
            pending_futures.add(future)
            future_meta[future] = (image_path, xml_path, text_path)

            while len(pending_futures) >= args.max_concurrency:
                done_futures, pending_futures = wait(
                    pending_futures,
                    return_when=FIRST_COMPLETED,
                )
                for done_future in done_futures:
                    image, xml_output, text_output = future_meta.pop(done_future)
                    try:
                        raw_output = done_future.result()
                        _save_outputs_for_image(
                            image_path=image,
                            xml_output_path=xml_output,
                            text_output_path=text_output,
                            model_label=args.vllm_model,
                            system_message=args.system_message,
                            raw_output=raw_output,
                        )
                        success += 1
                        print("[ok] image={} -> {}, {}".format(image, xml_output, text_output))
                    except Exception as exc:
                        failed += 1
                        print("[error] image={}: {}".format(image, exc))

        while pending_futures:
            done_futures, pending_futures = wait(
                pending_futures,
                return_when=FIRST_COMPLETED,
            )
            for done_future in done_futures:
                image, xml_output, text_output = future_meta.pop(done_future)
                try:
                    raw_output = done_future.result()
                    _save_outputs_for_image(
                        image_path=image,
                        xml_output_path=xml_output,
                        text_output_path=text_output,
                        model_label=args.vllm_model,
                        system_message=args.system_message,
                        raw_output=raw_output,
                    )
                    success += 1
                    print("[ok] image={} -> {}, {}".format(image, xml_output, text_output))
                except Exception as exc:
                    failed += 1
                    print("[error] image={}: {}".format(image, exc))

    return seen, success, failed, skipped


def main() -> int:
    args = _parse_args()
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.backend == "transformers" and args.max_concurrency != 1:
        print(
            "[warn] --max-concurrency only applies to --backend vllm. "
            "Transformers backend runs sequentially."
        )

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist: {}".format(args.input_dir))

    image_paths = _iter_input_images(args.input_dir)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print("[done] No images found under {}".format(args.input_dir))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        "[run] backend={} images={} input_dir={} output_root={}".format(
            args.backend, len(image_paths), args.input_dir, args.output_root
        )
    )

    if args.backend == "vllm" and args.max_concurrency > 1:
        seen, success, failed, skipped = _run_vllm_with_concurrency(args, image_paths)
        print(
            "[done] seen={} success={} failed={} skipped={} output_root={}".format(
                seen, success, failed, skipped, args.output_root
            )
        )
        return 1 if failed > 0 else 0

    device = None  # type: Optional[torch.device]
    processor = None  # type: Optional[AutoProcessor]
    model = None  # type: Optional[AutoModelForImageTextToText]

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

    seen = 0
    success = 0
    failed = 0
    skipped = 0

    for image_path in image_paths:
        seen += 1
        xml_path, text_path = _build_output_paths(args.output_root, args.input_dir, image_path)

        if args.skip_existing and xml_path.exists() and text_path.exists():
            skipped += 1
            print("[skip] image={} outputs exist".format(image_path))
            continue

        image = None
        try:
            image = _load_image(image_path)
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
                model_label = args.vllm_model
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
                model_label = args.model_id

            _save_outputs_for_image(
                image_path=image_path,
                xml_output_path=xml_path,
                text_output_path=text_path,
                model_label=model_label,
                system_message=args.system_message,
                raw_output=raw_output,
            )
            success += 1
            print("[ok] image={} -> {}, {}".format(image_path, xml_path, text_path))
        except Exception as exc:
            failed += 1
            print("[error] image={}: {}".format(image_path, exc))
        finally:
            if image is not None:
                image.close()

    print(
        "[done] seen={} success={} failed={} skipped={} output_root={}".format(
            seen, success, failed, skipped, args.output_root
        )
    )
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
