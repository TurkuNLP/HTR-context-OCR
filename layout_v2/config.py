"""layout_v2 configuration: every pinned constant in one auditable place.

Single source of truth for the whole pipeline. Nothing else in layout_v2 hardcodes a model id,
a resolution number, a vocabulary, or a threshold — if you need to change behaviour, change it
here and the change is visible in every run's ``run_config.json`` snapshot.

Decision provenance: IMPLEMENTATION_PLAN.md §14 (all items are project decisions, 2026-07-06).
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Dataset (pinned so results stay joinable with the existing Churro transcription runs)
# --------------------------------------------------------------------------------------
DATASET_ID = "stanford-oval/churro-dataset"
DATASET_REVISION = "ead84ec361300cf139969c93e058233a39da0970"  # legacy schema, matches old runs
DATASET_SPLITS = ("dev", "test")

# --------------------------------------------------------------------------------------
# Models (decision #15: these two ONLY; exact repo ids, never aliases)
# --------------------------------------------------------------------------------------
# Keyed by the short MODEL_KEY used by the launcher. ``serve_extra`` holds the model-specific
# vLLM startup toggles: the 35B-A3B family hangs after weight load without them (old job 120028).
MODELS: dict[str, dict] = {
    "thinking": {
        "repo": "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "reasoning_parser": "qwen3",   # vLLM separates the <think> trace into reasoning_content
        "serve_extra": [],             # starts fine with vLLM defaults
    },
    "qwen35": {
        "repo": "Qwen/Qwen3.5-35B-A3B",
        "reasoning_parser": "qwen3",
        "serve_extra": ["--enforce-eager", "--gdn-prefill-backend", "triton"],
    },
}

# Sampling per Qwen guidance for thinking-style models (decision #6: no greedy anywhere).
TEMPERATURE = 0.6
TOP_P = 0.95

# Per-pass generation caps. NOTE: vLLM's max_tokens covers the WHOLE generation including the
# thinking trace, so these are trace-inclusive budgets. The structured answers themselves are
# tiny (tens to ~300 tokens); the headroom is for reasoning. Small caps are the structural fix
# for the old pipeline's truncation-loss failure mode (35k cap, multi-page traces).
PASS0_MAX_TOKENS = 3000   # category + parts: short reasoning, tiny answer
PASS1_MAX_TOKENS = 9000   # column enumeration: the hard task, gets the largest budget
PASS1_LENGTH_RETRY_MAX_TOKENS = 14000  # rare rescue when pass 1 hits finish_reason=length
PASS2_MAX_TOKENS = 9000   # item enumeration / sampling; trace-inclusive, so sampling needs headroom
PASS2_LENGTH_RETRY_MAX_TOKENS = 14000  # rare rescue when pass 2 spends the cap before JSON content

# --------------------------------------------------------------------------------------
# Imaging policy (plan §5; decision #12: HF images at native size, capped by area only)
# --------------------------------------------------------------------------------------
MAX_FULL_PAGE_MP = 8.0        # cap on processed megapixels for the full-page image
JPEG_QUALITY = 92             # uploads are JPEG (PNG inflates newsprint payloads ~3x for no gain)
N_BANDS = 2                   # full-width horizontal band crops sent alongside the full page
BAND_HEIGHT_FRAC = 0.22       # each band is this fraction of page height, at native width
BAND_CENTERS = (0.45, 0.75)   # default band centers (fraction of page height), masthead avoided
MASTHEAD_SKIP_FRAC = 0.12     # never place a band above this fraction (masthead/title zone)
BAND_MAX_MP = 4.0             # per-band megapixel cap (native width usually stays below this)

# vLLM serving-side knobs the launcher must set explicitly (plan §10): the resolution policy is
# part of the config, not an implicit processor default. max_pixels is in PIXELS (w*h).
VLLM_MM_MAX_PIXELS = int(MAX_FULL_PAGE_MP * 1_000_000)
VLLM_LIMIT_MM_PER_PROMPT = 4  # full page + up to 2 bands + headroom
VLLM_MAX_MODEL_LEN = 32768    # largest request ~ full page + 2 bands + prompt + trace

# --------------------------------------------------------------------------------------
# Pass 0: category + parts (the shared frame)
# --------------------------------------------------------------------------------------
# Closed category vocabulary (decision: closed because it gates downstream questions).
CATEGORIES = (
    "newspaper",
    "periodical",
    "book",
    "letter",
    "manuscript",
    "register",
    "form",
    "map_or_plate",
    "other",
)
# Per-part content classes: drive pass-2 routing and the stream-existence expectations of O8.
CONTENT_CLASSES = ("running_text", "items_field", "mixed", "image_or_decoration")
# Writing direction per part (decision #14: vertical lines are LINES, not columns).
WRITING_DIRECTIONS = ("horizontal", "vertical", "mixed")

# Categories for which pass 2 (item counting) is a well-posed question (plan §8 gating).
ITEM_GATED_CATEGORIES = ("newspaper", "periodical", "register")

# --------------------------------------------------------------------------------------
# Pass 2: item counting
# --------------------------------------------------------------------------------------
ITEM_ENUM_MAX = 15  # <= this: enumerate every item; above: sampling arithmetic (decision #13)

# --------------------------------------------------------------------------------------
# Harness thresholds (validate.py / derive.py)
# --------------------------------------------------------------------------------------
MIN_COL_GAP_FRAC = 0.03      # adjacent column centers closer than this = validator issue
TILING_TOLERANCE = 0.25      # |unit_width * n_cols - block_width| / block_width must be <= this
# Category -> maximum column count that does NOT raise a prior-violation flag (plan §9.2).
# The prior never overrides enumeration; it only marks the record for review.
CATEGORY_COLUMN_PRIOR_MAX = {
    "book": 2,
    "letter": 1,
    "manuscript": 2,
    "form": 3,
    "map_or_plate": 1,
}

# --------------------------------------------------------------------------------------
# Analysis defaults (decision #11: default join target, overridable via CLI)
# --------------------------------------------------------------------------------------
DEFAULT_NLS_OUTPUTS = (
    "/scratch/project_2017385/dorian/Churro_copy/results/"
    "custom_churro_infer_dev_full1170_run1/vllm/dev/outputs.json"
)
# Column-count bins used by the analysis (threshold shape of the effect; ±1 noise tolerated).
COLUMN_BINS = ((1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 10_000, "7+"))


def column_bin(count: int | None) -> str:
    """Map a column count onto the analysis bins ('' for missing/invalid counts)."""
    if not isinstance(count, int) or count < 1:
        return ""
    for lo, hi, label in COLUMN_BINS:
        if lo <= count <= hi:
            return label
    return ""
