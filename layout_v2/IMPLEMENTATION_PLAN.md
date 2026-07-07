# layout_v2 — Implementation plan

**Goal.** Build a new, self-contained layout-annotation pipeline that measures page structure
(columns, independent parts, articles/sections, category-gated layout type) accurately enough to
**prove that layout and page complexity influence transcription quality** on the Churro dataset.

**Ground rules.**

1. `/scratch/project_2017385/dorian/qwen3vl_layout/` is **read-only reference**. Nothing in it is
   modified, imported, or assumed correct. Small helpers we want are *re-implemented or copied*
   into `layout_v2/`, never imported across.
2. The normative methodology is `qwen3vl_layout/COLUMN_COUNT_METHOD.md` (referenced below as
   **CCM**, its section numbers cited as CCM §N / operations as O1–O9). `LAYOUT_VS_LANGUAGE_STUDY.md`
   is ignored entirely.
3. Everything new lives in `/scratch/project_2017385/dorian/layout_v2/`.
4. Nothing in this plan is implemented yet; each phase below states its deliverables, checks, and
   gates.

---

## Table of contents

1. [What we carry over from the old pipeline — and what we deliberately do not](#1-carryover)
2. [Architecture overview: three passes + harness](#2-architecture)
3. [Directory layout — every file and its responsibility](#3-directory)
4. [Phase A — the resolution audit (offline, first, blocks attribution)](#4-phase-a)
5. [Phase B — imaging policy: full-size input and band crops](#5-phase-b)
6. [Pass 0 — category + parts (the shared frame)](#6-pass0)
7. [Pass 1 — column counting (CCM operationalized)](#7-pass1)
8. [Pass 2 — items: articles / advertisements / entries (category-gated)](#8-pass2)
9. [Harness: derivation, validation, and the per-document record](#9-harness)
10. [Serving and launcher changes](#10-serving)
11. [Fixtures, bake-offs, and acceptance gates](#11-fixtures)
12. [Phase F — the proof analysis: layout vs transcription quality](#12-analysis)
13. [Rollout order, effort, risks](#13-rollout)
14. [Decision log: defaults adopted, confirmations needed](#14-decisions)

---

<a name="1-carryover"></a>
## 1. What we carry over from the old pipeline — and what we deliberately do not

The old code is a good *systems* reference and a rejected *task design*. Reading
`layout_infer.py` / `run_layout_infer.sh` establishes these reusable facts and patterns:

**Carry over (as patterns, re-implemented in `layout_v2/`):**

| Pattern | Where it lives in the old code | Why it stays |
|---|---|---|
| Pinned dataset revision `ead84ec361300cf139969c93e058233a39da0970`, streaming load | `layout_infer.py` (DEFAULT_DATASET_REVISION, `load_dataset(..., streaming=True)`) | joinability with the existing transcription runs; no multi-GB materialization |
| OpenAI-compatible vLLM client with `guided_json` / `response_format` structured modes | `_structured_output_fields`, `request_layout` | proven to work against the CSC vLLM build |
| Bounded-concurrency submit/drain loop | `run_split` | overlaps HTTP latency, never holds the dataset |
| Retry only on empty/unparseable content, with backoff | `_is_retryable_parse_error` | right failure taxonomy |
| Run-numbered output dirs + full `run_config.json` snapshot | `resolve_next_run_path`, `main` | reproducibility; already saved our bacon once |
| Per-doc self-describing records incl. `raw_content`, `reasoning_content`, **`response_metadata.usage`** | `process_example` | usage tokens are the input to the resolution audit (Phase A) |
| Defensive JSON parsing (fence-stripping, first-object fallback) | `parse_layout_json` | costs nothing |
| Gold reference fields extracted for later analysis, never sent to the model | `gold_xml.extract_gold_fields` | needed by the analysis join; copy the module |
| SLURM launcher shape: serve → readiness wait → run → cleanup | `run_layout_infer.sh` | proven on Mahti GH200 nodes |

**Deliberately not carried over:**

1. The **flat 14-field schema and its prompt** (`layout_schema.py`) — replaced wholesale by the
   three-pass design (CCM §8–10). No confidence enums, no 1–5 scales, no `reading_order_*`, no
   `has_multiple_articles`, no free-prose `notes`.
2. The **single-call-does-everything** task structure — replaced by pass 0/1/2 with harness gating.
3. The unexamined `MAX_IMAGE_DIM = 2500` constant — replaced by an explicit, audited resolution
   policy (Phase A/B).
4. `--limit-mm-per-prompt '{"image":1}'` — the new design sends full page + band crops in one
   request; the launcher must allow ≥3 images per prompt.
5. The old `summarize()` cross-tabs (built around removed fields) — replaced by summaries over the
   new primitives.

---

<a name="2-architecture"></a>
## 2. Architecture overview: three passes + harness

Per CCM §8–10 and the task-separation decision (attention concentration; per-task inputs;
failure independence; per-task freezes):

```
                       ┌───────────────────────────────────────────────┐
                       │ Pass 0  STRUCTURE   (full page, moderate res) │
                       │  → document_category, parts[] (anchored),     │
                       │    per-part content_class                     │
                       └───────────────┬───────────────────────────────┘
              harness gate: category & parts injected as text into later passes
                    ┌──────────────────┴──────────────────────┐
                    ▼                                         ▼
   ┌───────────────────────────────────┐    ┌────────────────────────────────────────┐
   │ Pass 1  COLUMNS (exclusive task)  │    │ Pass 2  ITEMS (category-gated)         │
   │ input: full page + band crops     │    │ newspaper → articles, advertisements   │
   │ output: per-part column lists     │    │ register  → entries                    │
   │  (anchors, routes O3/O7/O8,       │    │ book/letter/manuscript → NOT RUN       │
   │   counting band, reconciliation)  │    │ output: per-part item lists / samples  │
   └───────────────┬───────────────────┘    └───────────────┬────────────────────────┘
                   └──────────────────┬─────────────────────┘
                                      ▼
                       ┌──────────────────────────────────┐
                       │ HARNESS  derive.py + validate.py │
                       │  counts = len(lists); flags;     │
                       │  sampling arithmetic; priors;    │
                       │  reconciliation verdicts         │
                       └──────────────┬───────────────────┘
                                      ▼
                       per-document record → layout_outputs.json
                                      ▼
                       analyze_vs_transcription.py  (the proof)
```

Principles enforced everywhere (CCM §6, §8, §10):

- **Primitives only**: the model emits enumerations and evidence; the harness computes every
  count, boolean, and aggregate.
- **Evidence before conclusion**: schema property order puts lists before any integer.
- **Earned uncertainty**: route disagreement flags (computed by harness from the model's route
  outputs) are the only uncertainty signal; no confidence fields exist.
- **Trivial path**: pass 0's category+parts lets ~80% of the corpus (books, letters, manuscripts)
  exit pass 1 in a few tokens and skip pass 2 entirely.
- Passes 1 and 2 can run in parallel per document (both depend only on pass 0); the sampling
  arithmetic that combines them happens in the harness afterwards.

---

<a name="3-directory"></a>
## 3. Directory layout — every file and its responsibility

```
layout_v2/
├── IMPLEMENTATION_PLAN.md        this file
├── README.md                     quickstart: how to run each phase
├── config.py                     ALL pinned constants in one place:
│                                   dataset id + revision; EXACT model repo ids
│                                   (never the alias — e.g. "Qwen/Qwen3-VL-30B-A3B-Thinking");
│                                   resolution policy numbers; band-crop geometry;
│                                   enumeration→sampling threshold; category vocabularies.
├── passes/
│   ├── pass0_structure.py        PROMPT + JSON SCHEMA + postprocess for pass 0
│   ├── pass1_columns.py          PROMPT + JSON SCHEMA + postprocess for pass 1 (CCM O1–O9)
│   └── pass2_items.py            PROMPT + JSON SCHEMA + postprocess for pass 2
├── imaging.py                    image acquisition + resize policy + band-crop generation
│                                   + data-URL encoding (JPEG, quality-pinned — not PNG; see §5)
├── client.py                     vLLM chat client: request/retry/structured-mode/usage capture
│                                   (re-implementation of the old request_layout pattern)
├── runner.py                     orchestrator: stream dataset → per-doc pass 0 → gate →
│                                   pass 1 (+ pass 2 if gated in) → harness → write record;
│                                   bounded concurrency; resume via --skip-existing;
│                                   run-numbered output dirs; run_config.json snapshot
├── derive.py                     harness derivations (counts, page-level conventions,
│                                   sampling arithmetic, reconciliation verdicts, prior flags)
├── validate.py                   mechanical validators (anchor distinctness, extent
│                                   monotonicity/tiling, cross-pass parts consistency,
│                                   category-gated vocabulary conformance)
├── gold_ref.py                   copy of gold_xml.py's extraction (reference-only fields
│                                   for analysis; never model input)
├── audit_resolution.py           Phase A: offline audit over OLD results/ dirs + new runs
├── fixtures.py                   frozen canonical pages + expected values (CCM §12) +
│                                   the fixture scorer (exact / ±1 / bin / structure match)
├── bakeoff.py                    anchor-format and input-strategy comparisons on fixtures
│                                   + a stratified 50-page dev sample
├── analyze_vs_transcription.py   Phase F: join layout metrics to transcription scores;
│                                   tables, plots, regression (the proof deliverable)
├── run_layout_v2.sh              SLURM launcher (adapted; §10)
├── fixtures/                     cached fixture images + expected-value JSON
├── results/                      run outputs (run-numbered, as before)
└── logs/
```

Nothing imports from `qwen3vl_layout/`. `gold_ref.py` is a copied file with a header comment
stating its origin.

---

<a name="4-phase-a"></a>
## 4. Phase A — the resolution audit (offline, first, blocks attribution)

**Why first.** CCM §2/§3.4: until we know what pixel sizes the server actually processed, no
improvement or regression is attributable to prompt vs optics.

**Key realization:** the old runs already stored `response_metadata.usage` per document, and the
old driver resized to long edge ≤ 2500 client-side. So the audit needs **no new GPU time**:

`audit_resolution.py` will:

1. Walk one or more old run dirs (`qwen3vl_layout/results/*/dev/responses/*.json`, read-only).
2. For each record: `prompt_tokens` from usage; subtract the (constant) text-token count of the
   system prompt + chat template (measured once with the tokenizer); the remainder ≈ image tokens.
3. Convert image tokens → processed pixels (× per-token tile area for the exact model, taken from
   its HF processor config — verified empirically in step 4, not assumed).
4. **Calibration check:** send 3 synthetic images of known sizes (e.g. 1000×1000, 2000×2500,
   800×600) through the same request path once and confirm the token↔pixel relation, pinning the
   tile size and any min/max clamping actually in effect.
5. Emit `audit_report.md`: per-run distribution of processed megapixels; the specific values for
   the canonical fixture pages; verdict per CCM §3.4 — *optics were/were not binding in past runs*.

**Gate A:** the verdict decides how much of Phase B is mandatory (input upscaling vs band crops
only) and calibrates expectations for the first fixture smoke test.

Deliverable: `audit_report.md` + the calibrated tokens↔pixels constants written into `config.py`.

---

<a name="5-phase-b"></a>
## 5. Phase B — imaging policy: full-size input and band crops

Implements CCM §3 as code in `imaging.py`:

1. **Source selection.** Default input = the HF dataset image (pinned revision) at native size.
   `config.py` records the policy: resize only if the processed-pixel estimate exceeds
   `MAX_MP_FULL_PAGE` (default 8 MP — above the demonstrated ~3 MP sufficiency bound of CCM §3.2,
   below the 25 MP dilution/cost regime of CCM §3.3). No blanket long-edge-2500 constant.
2. **Pixels-per-gutter targeting.** For pass 1, the effective constraint is gutters ≥ ~1–2 tiles
   (CCM §3.2). Rather than trying to detect gutters client-side, the design guarantees it
   geometrically with **band crops**:
   - `make_bands(image, parts_hint)` produces up to `N_BANDS` (default 2) full-width horizontal
     strips, each `BAND_HEIGHT_FRAC` (default 0.22) of the page height, positioned to avoid the
     masthead (top ~12%) and to sample different vertical regions (defaults: centers at 45% and
     75% of page height; if pass 0 reported parts, one band per major part instead).
   - Bands are sent at native width — a 0.22-height band of a 5 MP page is ~1.1 MP, so two bands +
     the full page stay well inside budget while giving each gutter ~5× the pixel density in the
     bands.
3. **Encoding.** JPEG (quality 92) data URLs, not PNG — the old PNG re-encode inflates payloads
   ~3× for scanned newsprint at no quality gain that survives the processor resize. Pinned in
   `config.py`; revisit only if artifacts appear on fine gutters (checked on fixtures).
4. **Multi-image request shape.** Pass 1's user turn = [full page, band 1, band 2] with a text
   part naming them ("image 1: full page; image 2: band at ~45% height; …"), so the prompt can
   instruct: *parts and spanning elements from image 1; enumerate and verify columns in the
   bands* (O4/O5 become partially input-enforced).

Deliverable: `imaging.py` + fixture-page contact sheets (full + bands) for eyeball verification.

---

<a name="6-pass0"></a>
## 6. Pass 0 — category + parts (the shared frame)

**Input:** full page only, moderate resolution (this task needs global structure, not gutters).

**Task (prompt content requirements):**

- Decide `document_category` from a **small closed vocabulary**:
  `newspaper | periodical | book | letter | manuscript | register | form | map_or_plate | other`.
  (Closed because it gates everything downstream; `other` is the escape hatch. The old free-text
  category produced synonyms the gate cannot use.)
- Enumerate the **independent parts** (O2): regions read as separate streams that do not flow into
  one another. Operational test stated in the prompt: *a part boundary exists where no text stream
  crosses* (full-width rule + content-class change are evidence, not the definition). Items
  sitting side by side in one band are NOT parts.
- For each part: a positional anchor (top/bottom fraction of page height), a 2–4-word content
  anchor, and a `content_class` from
  `running_text | items_field | mixed | image_or_decoration` — this drives pass 2 routing and
  O8's stream-existence expectations.
- Explicit statement in the prompt that most pages have exactly one part, and a one-part answer is
  expected to be short (the trivial path must be *cheap*, CCM §9).

**Output schema (sketch; property order = emission order):**

```json
{
  "document_category": "newspaper",
  "parts": [
    {"top_frac": 0.05, "bottom_frac": 0.72, "anchor": "news columns", "content_class": "running_text"},
    {"top_frac": 0.72, "bottom_frac": 1.0,  "anchor": "book-format insert", "content_class": "running_text"}
  ]
}
```

No counts asked; `independent_parts = len(parts)` is derived (CCM §8.2).

**Budget:** max_tokens ≈ 300. Expected output for 80% of corpus: category + one part ≈ 40 tokens.

**Vertical-script (CJK) rule — decided, no longer deferred:** CJK pages are written
top-to-bottom, and **a top-to-bottom text line is a LINE, not a column** (project decision;
supersedes CCM §4.8's deferral). Consequences: (a) pass 0 reports `writing_direction`
(`horizontal | vertical | mixed`) per part from visual evidence, cross-checkable against the
dataset's `main_script`; (b) pass 1's prompt carries the negative rule verbatim for vertical
parts, and the enumerable column unit for a vertical part is the **horizontal text register/band
(dan)** — side-by-side vertical lines within one register are lines of one unit, never columns;
a vertical part with one register has `columns = 1`; (c) the harness still marks
`vertical_script=true` so column-based analyses can be run with and without this stratum
(analysis stratification, not ontology deferral).

---

<a name="7-pass1"></a>
## 7. Pass 1 — column counting (CCM operationalized)

This is the centerpiece; every design element traces to a CCM operation.

**Input:** full page + band crops (§5.4), plus pass 0's parts injected as a short text preamble
("This page has 2 parts: 1) news columns, top 72%; 2) book-format insert, bottom 28%. Analyse each
part separately.").

**Task (prompt content requirements, mapped to CCM):**

1. Never state a count before listing evidence (O1; enforced structurally by schema order).
2. For each part (injected list — the model does not re-derive parts, it *uses* them; if it
   believes the parts are wrong it sets a `parts_disputed` flag rather than silently deviating —
   cross-pass consistency is adjudicated by the harness):
   - **Choose and name the counting band** (O4): which image / vertical range it counted in and
     why ("lower third; classified columns run unbroken").
   - **Enumerate the columns left-to-right with anchors** (O3): per column, `x_center_frac`
     (0–1 fraction of page width — the script-free positional anchor) **and** `anchor_text`
     (2–4 words where legible, empty string allowed for illegible scripts). Both anchors, always;
     the bake-off (§11) may later drop one.
   - **Width arithmetic** (O7): one clean column's width fraction + the part's text-block width
     fraction; the implied count. Always emitted (it is two numbers and a division).
   - **Stream check + returns** (O8): `stream_exists` (does one text flow column-to-column, with
     the hyphenation cue named in the prompt); if true, `returns` and the implied count; if false,
     the reason token `independent_items` (this doubles as the advertisements-regime signal).
   - **Spanning elements** (O6): the prompt states the domain fact — *display advertisements and
     banners occupy whole numbers of columns; their edges fall on gutter lines; use their edges as
     grid evidence* — and asks for `spanning_edges_consistent: true|false|none_present`.
   - **Cross-band check** (O5): `second_band_alignment: aligned|misaligned|not_checked` — with
     `misaligned` explicitly meaning "re-examine parts or flag".
3. Exclusions restated compactly (CCM §4): table/list/timetable sub-columns are not columns; a
   single line is not a column; **in vertical (CJK) writing each top-to-bottom line is a line,
   never a column — count horizontal registers (dan) instead**; masthead/margins ignored.
4. Trivial fast path stated: a single-block part is one list entry, no bands needed.

**Output schema (sketch; per part; property order is load-bearing):**

```json
{
  "parts": [
    {
      "part_index": 1,
      "counting_band": "lower half, image 3",
      "columns": [
        {"x_center_frac": 0.08, "anchor_text": "Lääkäreitä"},
        {"x_center_frac": 0.19, "anchor_text": "Alfred Holmström"}
      ],
      "width_check": {"unit_width_frac": 0.12, "block_width_frac": 0.97, "implied_count": 8},
      "stream": {"exists": false, "reason": "independent_items", "returns": null, "implied_count": null},
      "spanning_edges_consistent": true,
      "second_band_alignment": "aligned",
      "parts_disputed": false
    }
  ]
}
```

**What the model never outputs:** `column_count`. The harness derives it as `len(columns)`,
cross-checks against `width_check.implied_count` and `stream.implied_count`, and records the
reconciliation verdict (O9) — `routes_agree` or the per-route values (§9).

**Budget:** max_tokens ≈ 700 (nine columns with double anchors ≈ 250 tokens; thinking trace
extra — the cap applies to the final answer, with the trace budgeted separately per the serving
config). Sampling: temperature 0.6 / top_p 0.95 single sample (Qwen guidance for thinking-style
models); the 3-sample median variant is a bake-off arm, not the default.

---

<a name="8-pass2"></a>
## 8. Pass 2 — items: articles / advertisements / entries (category-gated)

**Gating (in the runner, not the model):** pass 2 runs only when pass 0's category is
`newspaper | periodical | register`. Books, letters, manuscripts: the article question is
ill-posed and never asked — their record gets `items: not_applicable`.

**Definitions baked into the prompt (defaults per the section-counting discussion; flagged for
confirmation in §14):**

- **Item, not rubric:** an article/notice is counted at its own heading **or dash-leader** start;
  rubric headings that merely group notices ("Uutisia Helsingistä") are *not* items themselves.
- **Continuation tails** (text concluding an item begun on a previous page, no marker on this
  page) are not counted.
- **The feuilleton/insert installment** is the content of its part, not an article of the news
  part; with parts injected from pass 0 this falls out of scoping automatically.
- **Advertisements are counted separately from articles** (they behaved differently in every page
  examined; conflating them was the old `section_count`'s main defect).
- Registers: the unit is the **entry**; same enumeration/sampling logic applies.

**Two counting modes, chosen by the model per part per item type (the enumeration→sampling
threshold `ITEM_ENUM_MAX = 15` lives in `config.py`):**

1. **Enumerate** (≤ threshold): list each item with a 2–4-word heading/incipit anchor. Count =
   `len(list)`, derived.
2. **Sample** (> threshold): pick one representative column (name it by its pass-1 `x_center_frac`
   or ordinal), count items in that column carefully, state
   `{"sampled_column": 4, "items_in_column": 9, "columns_with_items": 6}`. The **harness** —
   not the model — computes the estimate `items_in_column × columns_with_items`, using pass 1's
   validated column list to sanity-check `columns_with_items`. The model never multiplies.

**Input:** full page (items are large-scale marks: headings, rules, boxes); parts + column count
injected as text from passes 0/1 when available (runner may run pass 2 in parallel with pass 1 and
inject only parts — the arithmetic waits for both; see §9).

**Output schema (sketch):**

```json
{
  "parts": [
    {
      "part_index": 1,
      "articles": {"mode": "enumerate", "items": [{"anchor": "Kansakoulunmeno-sääntö"}, ...],
                   "sample": null},
      "advertisements": {"mode": "sample", "items": null,
                         "sample": {"sampled_column": 4, "items_in_column": 9, "columns_with_items": 6}}
    }
  ]
}
```

Derived by harness: `article_count`, `advertisement_count` (exact or `~estimate` with an
`is_estimate` flag), `entry_count` for registers. `has_multiple_articles` does not exist anywhere.

**Budget:** max_tokens ≈ 600.

---

<a name="9-harness"></a>
## 9. Harness: derivation, validation, and the per-document record

### 9.1 `validate.py` — mechanical checks (run before derivation; failures → flags, not crashes)

Per pass-1 part: `x_center_frac` strictly increasing; pairwise gaps > `MIN_COL_GAP_FRAC`
(≈0.03); anchors non-identical where non-empty; `unit_width_frac × len(columns)` within ±25% of
`block_width_frac` (tiling sanity); schema-conformant vocabulary per category (defense in depth —
the per-pass schemas already gate). Cross-pass: pass 2's `part_index`es ⊆ pass 0's;
`parts_disputed` propagated.

### 9.2 `derive.py` — the numbers (CCM §8.2: models emit primitives, harness computes)

Per part: `column_count = len(columns)`; reconciliation verdict from
{enumerated, width-implied, returns-implied}: `agree` (all present routes equal) /
`minor_disagree` (±1) / `disagree` (else) — with the values kept. Per page:
`independent_parts = len(parts)`; `column_count_dominant` = the part covering the largest
area fraction (the page-level convention, computed not asked); `article_count`,
`advertisement_count` (+ `is_estimate`); prior-violation flag (category=book & columns>2, etc.);
`vertical_script` routing flag; `needs_review` = any of {disagree, misaligned second band,
parts_disputed, validator failures}.

### 9.3 The per-document record (written by `runner.py`)

One JSON per document, mirroring the old record's good bookkeeping (raw content, reasoning,
usage, attempts, timestamps — per pass), plus: the three parsed pass outputs, every derived
quantity, every flag, and the gold-reference fields from `gold_ref.py`. Aggregated to
`layout_outputs.json` + `layout_summary.json` per split (summary rebuilt around the new
primitives: category distribution, parts histogram, per-category column histograms,
reconciliation-verdict rates, needs_review rate, per-pass token usage).

---

<a name="10-serving"></a>
## 10. Serving and launcher changes (`run_layout_v2.sh`)

Adapted from the old launcher (same module loads, readiness wait, cleanup), with these changes:

1. `--limit-mm-per-prompt '{"image":4}'` — full page + up to 2 bands (+ headroom).
2. **`--mm-processor-kwargs` explicitly set** from `config.py` (`max_pixels` per Phase A/B) — the
   resolution policy must be in the launcher, visible, not an implicit processor default.
3. `MODEL_REPO` pinned to exact repo ids in `config.py` — the two production models are
   `Qwen/Qwen3-VL-30B-A3B-Thinking` and `Qwen/Qwen3.5-35B-A3B`; `run_config.json` records repo id
   **and** resolved commit hash (`hf_hub` lookup) — the old runs' `vllm_model: "qwen3vl"` alias
   made runs unattributable. Note: the 35B-A3B family needed the old launcher's
   startup-reliability toggles (`ENFORCE_EAGER=1`, `GDN_PREFILL_BACKEND=triton`, per job 120028);
   the new launcher keeps both toggles and enables them by default for that model.
4. Per-pass `max_tokens` (300/700/600) instead of one 35k cap — the truncation-loss class of
   failure becomes structurally impossible.
5. Same GH200/gpumedium single-node shape; `MAX_MODEL_LEN` can drop to ~32k (largest request ≈
   full page + 2 bands ≈ 8–10k visual tokens + prompt + output) — smaller KV cache, more
   concurrency.
6. The runner makes 2–3 requests per document; `MAX_CONCURRENCY` stays request-level (default 8),
   so per-document passes pipeline naturally.

---

<a name="11-fixtures"></a>
## 11. Fixtures, bake-offs, and acceptance gates

### 11.1 Fixtures (`fixtures.py` + `fixtures/`) — frozen from CCM §12

| Page | Expected |
|---|---|
| `europeana_00675495` | 1 part; 9 columns; stream exists (returns 8); category newspaper |
| `newseye-fin_576474_0003` | 2 parts; part 1: 7 cols, stream; part 2: 2×1 (insert); category newspaper |
| `europeana_00675329` | 1 part; 8 columns (base grid); stream false for ad field; category newspaper |
| `europeana_00674544` | 2 parts (news + feuilleton); 3 columns; category newspaper |
| `europeana_00674591` | 1 part; 7–8 columns (cropped scan) — `minor_disagree` or review flag acceptable |
| `newseye-fin_576485` | ad mosaic; must yield `needs_review`/disagreement, **not** a confident wrong integer |
| + 4 trivial pages (book, letter, manuscript, register from dev) | 1 part; 1 column; pass 2 gated off (or entries for the register); output ≤ ~80 tokens |

The fixture scorer reports: exact / ±1 / bin correctness per part; structure match (parts count,
stream verdicts); flag behavior (fires on 576485/674591, silent on clean pages); token cost per
pass.

**Gate 1 (prompt iteration loop):** pass 1 prompt iterates against fixtures until the CCM §12
criteria hold on them. Cheap (10 pages × a few variants), fully local decision.

### 11.2 Bake-offs (`bakeoff.py`) — decided on fixtures + a 50-page stratified dev sample
### (no hand verification anywhere — per project decision)

Arms are compared **without any human labeling**, on four measurable axes:

1. **Fixture accuracy** — the 10 fixture pages have known truth (established in the method-doc
   sessions, already done; no new human effort).
2. **Internal-consistency rates** on the 50-page stratified sample: route-agreement
   (enumeration vs width-arithmetic vs returns), validator pass rate (monotonicity/tiling/anchor
   distinctness), cross-band alignment rate.
3. **Cross-arm agreement** — arms that agree with each other on pages where routes also
   internally agree are trusted; systematic divergence localizes the weaker arm via fixtures.
4. **Token cost.**

Bake-off arms: anchor format (dual default vs x-only vs text-only, non-Latin pages included to
stress text anchors); input strategy (full+2 bands vs full-only vs bands-only); **model:
`Qwen/Qwen3-VL-30B-A3B-Thinking` vs `Qwen/Qwen3.5-35B-A3B` — these two ONLY (project decision;
both already in the HF cache; no Instruct arm anywhere)**. Sampling per Qwen guidance for
thinking-style models: temperature 0.6 / top_p 0.95 single-sample baseline; 3-sample median as an
optional cost/accuracy arm; no greedy arm.

**Gate 2:** winners frozen into `config.py`; from here the pass-1 design does not move.

### 11.3 Full-run acceptance (dev split) — label-free criteria

- Parse failures = 0 (per-pass caps make truncation impossible; any remaining failure is a bug).
- All fixture expectations reproduced exactly (incl. per-part structure on 576474; review flag
  fires on 576485/674591, silent on clean pages).
- **Route-agreement rate** on multi-column pages ≥ ~85%, and — the label-free undercount check —
  on pages where the width-arithmetic route implies 7+, the enumerated count matches it (no
  systematic enumeration < width-implied pattern, which is the undercount signature).
- `needs_review` rate nonzero, concentrated on ad-mosaic/cropped pages (verified by the flags'
  co-occurrence with `stream.reason=independent_items`, not by human review).
- Median output tokens on book/letter/manuscript pages < 120.

---

<a name="12-analysis"></a>
## 12. Phase F — the proof analysis: layout vs transcription quality

The pipeline exists to answer one question: **do layout/page-complexity measurements predict
transcription quality?** `analyze_vs_transcription.py` (fresh code, minimal and self-contained):

1. **Join.** `layout_outputs.json` ↔ per-document transcription scores
   (`Churro_copy/results/custom_churro_infer_dev_full1170_run1/vllm/dev/outputs.json`, key
   `normalized_levenshtein_similarity`; path configurable, multiple runs averageable) on the image
   basename — same join the old analysis used, re-implemented.
2. **Complexity measures (all derived, all model-free given the annotations):**
   `column_count_dominant` (binned 1 / 2–3 / 4–6 / 7+), `independent_parts`,
   `article_count`, `advertisement_count`, `total_items`, `category`, plus gold-side covariates
   from `gold_ref.py` (gold char count as the text-length control).
3. **Deliverables (dev first, test as confirmation):**
   - Per-category and per-column-bin NLS tables (printed vs handwritten separated).
   - Scatter/box plots: NLS vs column bin; NLS vs total items; per-language overlays.
   - **Spearman** correlations (rank-based; robust to the bounded metric), overall and within
     language.
   - A language-controlled regression: NLS ~ column_bin + items + parts + log(gold_chars) +
     language dummies — reported as the shift in language-coefficient spread with vs without the
     layout terms, i.e. *how much of the apparent language effect is layout*. (Fractional-logit
     GLM as the engine since NLS ∈ [0,1]; OLS as appendix check.)
   - The **within-Finnish contrast**: Finnish pages 2–3 cols vs 7+ cols — language held fixed by
     construction; the single most direct exhibit for the goal.
4. Excluded from analysis by default: `vertical_script` pages (deferred ontology),
   `needs_review` pages (sensitivity check: with vs without).

Output: `analysis/` CSVs + PNGs + a short `ANALYSIS.md` with the headline numbers.

---

<a name="13-rollout"></a>
## 13. Rollout order, effort, risks

**Order (each step gated by the previous):**

| # | Step | Effort | GPU |
|---|---|---|---|
| 1 | Phase A audit (`audit_resolution.py` over old results) | 0.5–1 day | none (calibration: minutes) |
| 2 | `config.py`, `imaging.py`, fixture contact sheets | 0.5 day | none |
| 3 | Pass 0 + pass 1 prompts/schemas; `client.py`, `runner.py` skeleton | 1–2 days | none |
| 4 | Fixture loop (Gate 1) on a live server | 0.5–1 day | ~1 GPU-hour bursts |
| 5 | Bake-offs (Gate 2), label-free scoring (§11.2) | 1 day | few GPU-hours |
| 6 | Pass 2 + harness (`derive.py`, `validate.py`) | 1 day | smoke only |
| 7 | Full dev run + acceptance (§11.3) | 0.5 day | ~1 node-shift |
| 8 | `analyze_vs_transcription.py` + ANALYSIS.md | 1 day | none |
| 9 | Test-split confirmation run | 0.5 day | ~1 node-shift |

**Risks & mitigations:**

- *xgrammar/guided-JSON property-order support:* the design relies on schema property order being
  emission order; verified in step 3 with a 5-request probe; fallback = `response_format`
  json_schema mode, second fallback = ordered prose-JSON prompt + client-side parse (the old
  parser defends this).
- *Multi-image prompts on the CSC vLLM build:* probe in step 3; fallback = bands-only input for
  pass 1 (full page seen only by pass 0).
- *Anchor fabrication:* validators + fixtures catch systematic cases; the 50-page hand-check
  quantifies it once.
- *Category errors gating pass 2 wrongly:* `other`/low-frequency categories default to running
  pass 2 in enumerate mode (over-asking is cheap; under-asking loses data); category confusion
  matrix reviewed at step 5.
- *Cost:* 2–3 calls/page ≈ 2.5–3.5k visual-token-dominated requests; dev split ≈ same order as one
  old run despite the extra passes (per-pass output caps are far below the old 35k).
- *CJK/vertical:* explicitly flagged and excluded from column analysis until the ontology decision
  (CCM §4.8) — no silent wrong numbers.

---

<a name="14-decisions"></a>
## 14. Decision log: defaults adopted, confirmations needed

Adopted as defaults in this plan (each traceable to the discussions):

1. Three passes with harness gating; passes 1–2 parallelizable per doc; arithmetic in harness.
2. Dual anchors (x_center_frac + short text) pending bake-off.
3. Items-not-rubrics; continuation tails uncounted; feuilleton installment ≠ article;
   `advertisement_count` separate; `ITEM_ENUM_MAX = 15`.
4. Closed category vocabulary (9 values + other).
5. JPEG q92 uploads; full page ≤ 8 MP processed; 2 bands × 0.22 height at native width.
6. Per-pass token caps 300/700/600; sampling = temperature 0.6 / top_p 0.95 (Qwen guidance for
   thinking-style models; no greedy arm).
7. No confidence fields anywhere; `needs_review` (computed) is the only uncertainty channel.
8. **No hand verification anywhere**: fixtures + label-free internal-consistency criteria replace
   all human labeling (§11.2–11.3).

All former open questions are now decided (project decisions, 2026-07-06):

9. **`advertisement_count`: in scope** for pass 2 (newspaper/periodical).
10. **Register `entry_count`: in scope** (same enumeration/sampling machinery).
11. **Analysis join target:** defaults to
    `Churro_copy/results/custom_churro_infer_dev_full1170_run1/vllm/dev/outputs.json`; any
    user-supplied `outputs.json` path(s) accepted via CLI (`--nls-outputs`, repeatable).
12. **Image source: HF dataset images** (pinned revision), native resolution, subject only to the
    §5 processed-pixel cap. The reduced local copies are not used as model input.
13. **Item definitions confirmed as stated:** items-not-rubrics; continuation tails uncounted;
    feuilleton installment ≠ article; `ITEM_ENUM_MAX = 15`.
14. **Vertical-CJK rule defined** (supersedes deferral): CJK writes top-to-bottom; a top-to-bottom
    text line is a line, never a column; the column unit for vertical parts is the horizontal
    register (dan); `vertical_script` flag retained for analysis stratification (§6).
15. **Models: `Qwen/Qwen3-VL-30B-A3B-Thinking` and `Qwen/Qwen3.5-35B-A3B` ONLY** (both cached);
    no Instruct; 35B startup toggles on by default (§10).
16. **Split order:** full dev first; test exactly once as the final confirmation run.
17. **`needs_review` pages excluded from the analysis by default**, with one sensitivity re-run
    including them.

No open questions remain; implementation can start at Phase A (§4).
