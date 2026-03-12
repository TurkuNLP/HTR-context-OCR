from __future__ import annotations

from dataclasses import dataclass
from time import time

from datasets import load_dataset

from churro.args import CHURRO_DATASET_ID, create_output_prefix
from churro.evaluation.metrics import compute_metrics
from churro.systems.ocr_factory import OCRFactory
from churro.utils.image.binarizer import ImageBinarizer
from churro.utils.llm.models import MODEL_MAP
from churro.utils.log_utils import logger

from .helpers import managed_vllm_container


VALID_DATASET_SPLITS = {"dev", "test"}


@dataclass(slots=True)
class BenchmarkOptions:
    system: str
    engine: str | None
    tensor_parallel_size: int
    data_parallel_size: int
    resize: int | None
    max_concurrency: int
    input_size: int
    dataset_split: str
    offset: int
    binarize: bool = False


def _validate_options(options: BenchmarkOptions) -> int:
    """Validate CLI options before any heavy work starts.

    Why this exists:
    - It fails fast on malformed inputs before dataset loading or model startup.
    - It centralizes policy checks (required engine, split validity, concurrency).
    """
    # Both `llm` and `finetuned` systems rely on MODEL_MAP-driven engine lookup.
    # Requiring `--engine` here prevents ambiguous runtime failures later.
    if options.system in {"llm", "finetuned"}:
        if not options.engine:
            logger.error(f"LLM engine must be specified for the '{options.system}' system.")
            return 1
        if options.engine not in MODEL_MAP:
            valid_engines = ", ".join(sorted(MODEL_MAP.keys()))
            logger.error(f"Invalid engine: {options.engine}. Possible values are: {valid_engines}")
            return 1
    if options.max_concurrency < 1:
        logger.error("--max-concurrency must be >= 1.")
        return 1
    # Observed in practice: finetuned OCR with high local concurrency can trigger
    # long-tail request timeouts on vLLM. Keep this as warning (not hard error)
    # so advanced users may still override intentionally.
    if options.system == "finetuned" and options.max_concurrency > 8:
        logger.warning(
            "High concurrency for finetuned OCR can trigger local vLLM timeouts. "
            "Consider --max-concurrency 2..8 for stability."
        )
    if options.dataset_split not in VALID_DATASET_SPLITS:
        valid = ", ".join(sorted(VALID_DATASET_SPLITS))
        logger.error(f"Invalid dataset split '{options.dataset_split}'. Choose from: {valid}.")
        return 1
    return 0


async def run(options: BenchmarkOptions) -> int:
    """Execute full benchmark workflow: validate -> load -> infer -> score."""
    validation_status = _validate_options(options)
    if validation_status != 0:
        return validation_status

    # Output prefix controls where metrics artifacts and per-example outputs land.
    output_prefix = create_output_prefix(options)  # type: ignore[arg-type]

    # Build an optional streaming slice:
    # - input_size==0 means "all remaining examples after offset"
    # - otherwise evaluate only [offset : offset+input_size]
    start_index = options.offset
    end_index = options.offset + options.input_size if options.input_size > 0 else None

    logger.info(
        f"Loading dataset slice: split={options.dataset_split}, offset={options.offset}, "
        f"limit={options.input_size if end_index is None else options.input_size}"
    )
    dataset = list(load_dataset(CHURRO_DATASET_ID, split=options.dataset_split, streaming=True))
    dataset = dataset[start_index:end_index]

    elapsed_time = 0.0
    binarizer: ImageBinarizer | None = None
    if options.binarize:
        logger.info("Binarizing dataset images prior to benchmarking.")
        try:
            binarizer = ImageBinarizer()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error(f"Failed to initialize image binarizer: {exc}")
            return 1
    with managed_vllm_container(
        engine=options.engine,
        backup_engine=None,
        system=options.system,
        tensor_parallel_size=options.tensor_parallel_size,
        data_parallel_size=options.data_parallel_size,
    ):
        # OCR system creation happens inside the managed context so any required
        # local vLLM resources are available before first request.
        ocr_system = OCRFactory.create_ocr_system(options)  # type: ignore[arg-type]
        start_time = time()
        images = [example["image"] for example in dataset]
        if binarizer is not None:
            try:
                images = binarizer.binarize_pil_batch(images)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error(f"Binarizer batch inference failed: {exc}")
                return 1
        predicted_texts = await ocr_system.process_images(
            images,
            max_concurrency=options.max_concurrency,
        )
        # Measure inference wall time only (not dataset loading + metric writing).
        elapsed_time = time() - start_time

    # Hard invariant: each input example must have exactly one prediction.
    assert len(dataset) == len(predicted_texts), (
        f"Mismatch in number of examples ({len(dataset)}) and predicted texts ({len(predicted_texts)})."
    )

    compute_metrics(dataset, predicted_texts, output_prefix, elapsed_time)
    return 0
