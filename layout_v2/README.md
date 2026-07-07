# layout_v2 — three-pass layout annotation for the Churro dataset

Measures page structure (document category, independent parts, per-part **column counts**,
articles / advertisements / register entries) with a VLM, to test whether **layout and page
complexity influence transcription quality**. Methodology:
`../qwen3vl_layout/COLUMN_COUNT_METHOD.md`; design + all project decisions:
`IMPLEMENTATION_PLAN.md` (this directory).

The old pipeline (`../qwen3vl_layout/`) is read-only reference; nothing here imports from it.

## How it works (one document)

```
pass 0  (full page)          -> document_category + independent parts        passes/pass0_structure.py
pass 1  (full page + bands)  -> per-part COLUMN ENUMERATION with anchors,    passes/pass1_columns.py
                                width-arithmetic + returns routes
pass 2  (full page)          -> articles/ads/entries, enumerate or sample    passes/pass2_items.py
                                (only for newspaper/periodical/register)
harness                      -> counts = len(lists); route reconciliation;   derive.py, validate.py
                                needs_review flags (the only uncertainty channel)
```

The model never outputs a count — every number is derived from its enumerations. Two models are
supported (`config.MODELS`): `thinking` (Qwen3-VL-30B-A3B-Thinking) and `qwen35`
(Qwen3.5-35B-A3B). All constants live in `config.py`.

## Run (on Mahti)

```bash
cd /scratch/project_2017385/dorian/layout_v2

# 0) Phase A: resolution audit over OLD runs (no GPU; do this once, first)
python3 audit_resolution.py \
  --runs ../qwen3vl_layout/results/layout_thinking_dev_1GPU_v4_run1/dev \
  --report audit_report.md
# optional exact calibration against a live server:
#   python3 audit_resolution.py --calibrate --vllm-base-url http://localhost:8000/v1 --vllm-model layoutv2

# 0b) Preflight probes against a live server (multi-image, property order, calibration).
#     Run once per vLLM build; needs a server up (e.g. during any interactive/fixture job):
python3 preflight.py --vllm-base-url http://localhost:8000/v1 --vllm-model layoutv2

# 1) Fixture smoke run (the 6 canonical pages; Gate 1)
ONLY_BASENAMES="$(python3 fixtures.py --list)" RUN_LABEL=fixtures sbatch run_layout_v2.sh
python3 fixtures.py --run results/layout_v2_fixtures_run1/dev

# 2) Small smoke (first 10 docs)
MAX_SAMPLES_PER_SPLIT=10 RUN_LABEL=smoke sbatch run_layout_v2.sh

# 2b) Bake-off arms (Gate 2): fixture pages + a slice, one flag flipped per arm, then compare.
FIX="$(python3 fixtures.py --list)"
ONLY_BASENAMES="$FIX" RUN_LABEL=arm_dual                                   sbatch run_layout_v2.sh
ONLY_BASENAMES="$FIX" RUN_LABEL=arm_xonly  PASS1_ANCHOR_MODE=x_only        sbatch run_layout_v2.sh
ONLY_BASENAMES="$FIX" RUN_LABEL=arm_full   PASS1_INPUT=full_only           sbatch run_layout_v2.sh
ONLY_BASENAMES="$FIX" RUN_LABEL=arm_q35    MODEL_KEY=qwen35                sbatch run_layout_v2.sh
python3 bakeoff.py --runs results/layout_v2_arm_*_run1/dev --report bakeoff_report.md

# 3) Full dev, both models (with the Gate-2 winning arm flags)
RUN_LABEL=thinking sbatch run_layout_v2.sh
MODEL_KEY=qwen35 RUN_LABEL=qwen35 sbatch run_layout_v2.sh

# 4) The proof analysis (join to transcription NLS; no GPU)
python3 analyze_vs_transcription.py --layout-run results/layout_v2_thinking_run1/dev
# custom transcription scores:  --nls-outputs /path/to/outputs.json [more...]

# 5) Test split, exactly once, at the end (decision #16)
DATASET_SPLIT=test RUN_LABEL=test_confirm sbatch run_layout_v2.sh
```

## Outputs (per run, run-numbered)

- `results/<run>/run_config.json` — full config snapshot incl. exact model repo + commit hash.
- `results/<run>/<split>/responses/*.json` — one self-describing record per document
  (all three pass results with raw content, reasoning trace, token usage; parsed outputs;
  validator issues; every derived quantity and flag).
- `results/<run>/<split>/layout_outputs.json` — all records (the analysis input).
- `results/<run>/<split>/layout_summary.json` — distributions over the new primitives.
- `results/<run>/<split>/analysis/` — tables, plots, `ANALYSIS.md` (after step 4).

## Key project decisions (full log: IMPLEMENTATION_PLAN.md §14)

- No confidence fields, no 1–5 scales; `needs_review` is computed, never asked.
- No hand verification: fixtures + label-free internal-consistency criteria.
- Vertical (CJK) text: a top-to-bottom line is a LINE, never a column; vertical pages are flagged
  and excluded from column analyses by default.
- Sampling: temperature 0.6 / top_p 0.95 (thinking-model guidance); per-pass token caps.
- `has_multiple_articles` does not exist; `article_count >= 2` is derivable.
