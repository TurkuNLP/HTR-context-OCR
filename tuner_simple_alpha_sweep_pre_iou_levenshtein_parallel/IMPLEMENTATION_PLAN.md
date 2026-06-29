# Implementation plan — Parallel two-phase (scout + refine) alpha sweep

This directory will hold a **full standalone copy** of
`tuner_simple_alpha_sweep_pre_iou_levenshtein/` with one behavioral change: the per-document
alpha sweep becomes a **two-phase scout + refine** search evaluated across **`floor(cpus/2)`
worker processes**, instead of a serial loop over every alpha at full Hough-run count.

> Status when this doc was written: only this plan file exists in the directory. The actual
> package copy + import-rename + Cython build were **blocked** because `/tmp` was full
> (`ENOSPC`) and the shell could not run. Execute "Step 1 — Copy" below once the shell is back.

---

## Why

The alpha sweep dominates runtime. Per document it issues
`num_alphas × hough_num_runs` calls to `probabilistic_hough_line`
(e.g. 35 alphas × 10 runs = **350** Hough calls), entirely serial — even though the alpha
candidates are mutually independent and the bulk cost is the `N`-run Hough union per alpha.

Two independent speedups, combined:

1. **Two-phase scout + refine.** Scan *every* alpha cheaply with a small number of Hough runs
   (`scout`), rank them, then re-run only the top-`K` alphas at the full `hough_num_runs`
   (`refine`) and select the winner from those.
2. **Process-level parallelism.** Evaluate alpha candidates concurrently using
   `W = max(1, cpus // 2)` worker processes (user-specified ratio: leaves headroom and avoids
   oversubscription against BLAS/skimage internals and the dynamic-worker document pool).

`--alpha-sweep-step` already exists (default 0.2) and is the coarse speed dial; raising it is a
third, orthogonal lever that needs no code change.

**Balanced defaults** (all CLI-overridable): `scout_runs = 3`, `refine_top_k = 5`,
`workers = floor(cpus/2)`.

Worked example, 18 alphas (step 0.2), `N=10`:
`18×3 (scout) + 5×10 (refine) = 104` Hough calls vs `180` serial — then divided by `W`
in wall-clock. The **selected** candidate is always a full-`N` refine candidate, so it is
bit-identical to what the original pipeline computes for that alpha; the only deviation from
exhaustive search is the (small) chance the scout pass ranks the true-best alpha outside the
top-5.

---

## Step 1 — Copy (do this first, needs a shell)

```bash
cd /scratch/project_2017385/dorian/Churro_copy
SRC=tuner_simple_alpha_sweep_pre_iou_levenshtein
DST=tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel

# 1a. Recursive copy (preserve this plan file if DST already exists)
cp -rn "$SRC"/. "$DST"/        # -n: don't clobber IMPLEMENTATION_PLAN.md

# 1b. Mechanical import rename across every .py (long, unique token — safe)
grep -rl 'tuner_simple_alpha_sweep_pre_iou_levenshtein' "$DST" --include='*.py' \
  | xargs sed -i 's/tuner_simple_alpha_sweep_pre_iou_levenshtein/tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel/g'

# 1c. Remove any stale compiled artifacts copied from the source (force a clean rebuild)
find "$DST" -name '*.so' -delete
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} +
```

The package uses **absolute imports rooted at the package name**, so 1b is sufficient to
re-root the whole tree. Replacing the package name inside schema-version strings is harmless.

Subpackages carried over unchanged: `config/ document_selection/ matrix_operations/
probabilistic_hough/ scoring/ serial_runner/ results_writing/ dynamic_pool/ plotting/
logging_utils/ runtime/ cython_accel/` plus `run_tuner_simple.py`.

**Cython caveat:** `cython_accel/build.py` builds 3 extensions (`ownership_core`, `filter_core`,
`threshold_mask_core`) with **package-relative names** (`name="cython_accel.X"`). The entry
script must insert *this* package dir at `sys.path[0]` so `import cython_accel` resolves to the
copy's freshly built `.so` files. Never import both packages in one interpreter.

---

## Step 2 — Files that change (only these, in the copy)

| File | Change |
|---|---|
| `config/cli_arguments.py` | +4 CLI flags; thread into `PipelineConfig` |
| `config/pipeline_config.py` | +4 fields + validation |
| `serial_runner/document_runner.py` | Replace the alpha-sweep `else` branch (orig lines 1232–1301) with `run_alpha_sweep_parallel(...)`; add orchestrator + worker tasks + module globals — all in this file so `run_alpha_candidate`, `build_plot_payload`, `build_alpha_candidate_summary`, `candidate_selection_key` stay in scope (no circular import) |
| `run_tuner_simple.py` | Insert this package dir at `sys.path[0]` |
| `run_tunner_parallel.sh` (new) | Launcher modelled on `run_tunner.sh` |

`pipeline_runner.py` and `dynamic_worker_runner.py` are **unchanged** — both already call
`process_one_document`, which switches internally.

---

## Step 3 — CLI flags (`config/cli_arguments.py`)

Keep `--alpha-sweep-step` as-is. Add:

```python
parser.add_argument("--two-phase", dest="two_phase_enabled", action="store_true")
parser.add_argument("--no-two-phase", dest="two_phase_enabled", action="store_false")
parser.set_defaults(two_phase_enabled=True)
parser.add_argument("--scout-hough-runs", type=int, default=3,
    help="Hough runs per alpha in the scout pass (cheap ranking). Refine uses --hough-num-runs.")
parser.add_argument("--refine-top-k", type=int, default=5,
    help="How many top scout alphas are re-run at full --hough-num-runs and selected from.")
parser.add_argument("--alpha-parallel-workers", type=int, default=0,
    help="Worker processes for alpha candidates. 0 = auto = max(1, cpus//2).")
```

Thread the four values into `PipelineConfig(...)` in `parse_pipeline_config`.

---

## Step 4 — Config fields (`config/pipeline_config.py`)

```python
two_phase_enabled: bool = True
scout_hough_runs: int = 3
refine_top_k: int = 5
alpha_parallel_workers: int = 0      # 0 => auto floor(cpus/2)
```

In `validate()`: `scout_hough_runs >= 1`, `refine_top_k >= 1`, `alpha_parallel_workers >= 0`.
Do **not** raise when `scout_hough_runs >= hough_num_runs`; the runner auto-falls back to
single-phase and logs it (see Step 5, phase decision).

---

## Step 5 — The orchestrator (`serial_runner/document_runner.py`)

### 5a. Worker-count resolution

```python
def _detect_cpu_count() -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v:
        return int(v)
    try:
        return len(os.sched_getaffinity(0))   # respects cgroup / SLURM binding
    except Exception:
        return os.cpu_count() or 1

def resolve_alpha_worker_count(config) -> int:
    if config.alpha_parallel_workers > 0:
        return max(1, config.alpha_parallel_workers)
    return max(1, _detect_cpu_count() // 2)
```

### 5b. Read-only per-document context (fork + copy-on-write)

A module global holds the heavy read-only data so forked workers inherit it **without
pickling**. It is set in the parent immediately before pool creation and cleared in `finally`:

```python
_ALPHA_WORKER_CONTEXT: dict | None = None
```

Contents: `document, config, ref_to_pred_matrix, ref_to_ref_matrix, ref_to_pred_shape,
ref_to_ref_shape, ref_to_pred_floor_statistics, ref_to_ref_floor_statistics, the 4 matrix
source/reason strings, reference_windows, prediction_windows, timing_matrix_seconds,
document_normalised_levenshtein, keep_plot_payload`.

### 5c. Per-alpha effective config (per-phase run count)

`run_alpha_candidate` reads `hough_num_runs` from the config it is handed, so we only swap that
field — **`run_alpha_candidate` itself is not modified**:

```python
def _effective_config(base_config, *, num_runs):
    return replace(base_config,
        hough_parameters=replace(base_config.hough_parameters, hough_num_runs=num_runs))
```

### 5d. Worker tasks (module-level, picklable by reference)

```python
def _run_scout_task(alpha, num_runs):
    ctx = _ALPHA_WORKER_CONTEXT
    cand = run_alpha_candidate(alpha=alpha,
                               config=_effective_config(ctx["config"], num_runs=num_runs),
                               **_ctx_rest(ctx))
    return (alpha, candidate_selection_key(cand), build_alpha_candidate_summary(cand))

def _run_refine_task(alpha, num_runs):
    ctx = _ALPHA_WORKER_CONTEXT
    cand = run_alpha_candidate(alpha=alpha,
                               config=_effective_config(ctx["config"], num_runs=num_runs),
                               **_ctx_rest(ctx))
    key = candidate_selection_key(cand)
    summary = build_alpha_candidate_summary(cand)
    plot_bits = None
    if ctx["keep_plot_payload"]:
        # only what build_plot_payload needs beyond the matrices (parent already holds those)
        plot_bits = (
            cand.ref_to_pred_floor.hough_input_mask,
            list(cand.ref_to_pred_hough.detection_result.get("raw_lines", [])),
            list(cand.scored.ref_to_pred_payload.hough_payload.filtered_result.get("lines_used", [])),
        )
    return (alpha, key, summary, plot_bits)
```

Scout returns are tiny (tuple + summary dict). Refine returns add only a boolean mask + two
short line lists, and only when plotting is on. **No score matrices and no full
`AlphaCandidateRun` ever cross the process boundary.**

### 5e. Fork-safety RNG init

```python
def _alpha_worker_init():
    import numpy as np
    np.random.seed()   # reseed each forked child from OS entropy
```

Needed only in **seed=None** (non-deterministic) mode so forked siblings don't share inherited
RNG state and their Hough unions stay diverse. When `--hough-seed` is set,
`run_alpha_candidate` passes an explicit per-run `np.random.default_rng(seed)` to skimage, so
determinism is unaffected and the parallel build is **bit-identical** to serial for a given
alpha + N.

### 5f. `run_alpha_sweep_parallel(...)` flow

1. **Empty-mask pre-scan** (serial, cheap). For each alpha compute `floor = mean + alpha*std`
   and `active = count(ref_to_pred_matrix >= floor)`. `active` is monotone non-increasing in
   alpha → find `first_empty`. `active_alphas = alphas[:first_empty]`. Evaluate the single
   `first_empty` alpha in-process (cheap empty-Hough path) and append its zero-metric summary —
   this reproduces the serial early-exit's recorded candidate set exactly.
2. **Phase decision.** Use single-phase (all `active_alphas` at full `N`, then select) when:
   `not two_phase_enabled` **or** `scout_hough_runs >= hough_num_runs` **or**
   `refine_top_k >= len(active_alphas)` **or** `len(active_alphas) <= 1`. Otherwise two-phase.
   Log which path was taken.
3. Set `_ALPHA_WORKER_CONTEXT`; resolve `W = resolve_alpha_worker_count(config)`.
4. **Dispatch.** If `W == 1`, call the task functions in-process (no pool). Else create
   `multiprocessing.get_context("fork").Pool(processes=W, initializer=_alpha_worker_init)`,
   reuse the one pool for both phases, and close it in `finally`.
   - Scout: `_run_scout_task` over every active alpha at `scout_hough_runs`.
   - Rank scout results by selection key (descending); take the top `refine_top_k` **alphas**.
   - Refine: `_run_refine_task` over those alphas at `hough_num_runs`.
5. **Winner** = max refine result by selection key (single-phase: max over the full-`N` run).
6. **Assemble outputs** in the parent:
   - `candidate_summaries`: **one per active alpha** — the refine summary where the alpha was
     refined, else its scout summary; each tagged `scout_phase ∈ {"scout","refine","single"}`
     and `hough_num_runs_used`. Append the first-empty summary if present.
   - `selected_result_row` = winner summary's `result_row`; `selected_candidate_summary` =
     winner summary.
   - `selected_plot_payload`: built in the parent from the winner's returned `plot_bits`
     (mask + raw_lines + final_lines) plus the matrices the parent already holds — assemble the
     same dict shape `build_plot_payload` produces.
7. Clear `_ALPHA_WORKER_CONTEXT` in `finally`.

### 5g. Integration point

In `process_one_document`, replace the alpha-sweep `else` block (original lines 1232–1301) with
a single call to `run_alpha_sweep_parallel(...)` returning
`(selected_result_row, selected_candidate_summary, selected_plot_payload, candidate_summaries)`.
Everything downstream (pickle payload, plot payload wiring, result row, CSV) is unchanged.

### 5h. Functions reused as-is (no behavior change)

- `run_alpha_candidate` (the parallel unit; called with an effective config)
- `candidate_selection_key`
- `build_alpha_candidate_summary`
- `build_plot_payload`
- `alpha_values_for_config` (honors `--alpha-sweep-step`)
- `compute_score_floor_statistics` / `compute_score_floor_mask_from_statistics`
  (`matrix_operations/score_floor.py`) — the pre-scan uses the same mean/std/floor math.

---

## Step 6 — Launcher (`run_tunner_parallel.sh`)

Copy `run_tunner.sh` into this dir, then:

- `#SBATCH --job-name=tuner_simple_parallel`; keep account/partition/time/nodes/ntasks/mem.
  `--cpus-per-task` is what drives `floor(cpus/2)` — document this in the usage text.
- Keep the `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` /
  `NUMEXPR_NUM_THREADS` `=1` exports **verbatim** — required for fork safety and so each alpha
  worker stays single-threaded.
- Point `SCRIPT_DIR` resolution + absolute fallback at this `_parallel` dir; entry stays
  `run_tuner_simple.py`.
- Keep the `cython_accel/build.py` build block (now builds this copy's extensions in place).
- Add `append_env_default` passthroughs: `SCOUT_HOUGH_RUNS → --scout-hough-runs`,
  `REFINE_TOP_K → --refine-top-k`, `ALPHA_PARALLEL_WORKERS → --alpha-parallel-workers`; honor
  `NO_TWO_PHASE=1 → --no-two-phase`. Extend the usage text accordingly.

---

## Edge cases

| Situation | Behaviour |
|---|---|
| `W == 1` (1–2 CPUs) | No pool; tasks run in-process. Still two-phase (still saves Hough calls). |
| `--no-two-phase` | All active alphas at full `N` (parallelized); selection identical to original. |
| `scout_runs >= N` or `top_k >= num_active` or `num_active <= 1` | Auto single-phase; logged. |
| First alpha already empties the mask | That zero candidate is the selection (matches serial). |
| `--hough-seed` set | Parallel output bit-identical to serial per alpha. |
| seed=None | Non-deterministic (as today); workers reseed so unions stay diverse. |
| Dynamic-worker mode | Each SLURM task forks its own `floor(cpus/2)` within its cgroup; no global oversubscription. |

---

## Verification

1. **Behavior-preservation (deterministic).** ~10 docs; run the ORIGINAL package and this copy
   with identical args **plus** `--hough-seed 123 --no-two-phase --alpha-parallel-workers 1`.
   Diff `best_combination_per_document.csv` on `score_floor_alpha` + the 6 metrics — must match
   exactly.
2. **Parallel equivalence.** As #1 but `--alpha-parallel-workers 4` — must still match #1
   (proves the pool/fork path does not alter results).
3. **Two-phase quality.** ~50 docs; original full sweep vs this copy with Balanced defaults +
   `--hough-seed 123`. Count docs differing in selected alpha / metrics (expect ~0, tiny
   deltas). Raise `--refine-top-k` if any material regression appears.
4. **Speed.** Time #3 both ways; confirm the expected ~2–3× fewer Hough calls × ~`W`
   wall-clock gain.
5. **Picklability / fork.** Run with `--alpha-parallel-workers 2` on a few docs; confirm no
   `PicklingError`, no deadlock, and `OMP_NUM_THREADS=1` in the worker env.
6. **Cython.** Confirm `cython_accel/*.so` build here and that
   `import cython_accel.threshold_mask_core` resolves to this copy (add a log line in the entry
   script).

Run from `/scratch/project_2017385/dorian/Churro_copy` via `run_tunner_parallel.sh`
(or `sbatch` it).
