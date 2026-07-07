# Resolution audit (Phase A)

Vision tile area used: **784 px/token** (default, uncalibrated).

| run | n | text tokens (assumed) | min MP | median MP | max MP |
|---|---|---|---|---|---|
| ../qwen3vl_layout/results/layout_qwen35_dev_retry_run1/dev | 1170 | 1588 | 0.5 | 3.53 | 5.12 |
| ../qwen3vl_layout/results/layout_qwen35_dev_run1/dev | 0 | - | - | - | - |

## Canonical fixture pages (processed MP per run)

**../qwen3vl_layout/results/layout_qwen35_dev_retry_run1/dev**
- europeana_00674591: 3.9 MP
- newseye-fin_576485_0001_23676428: 4.02 MP
- newseye-fin_576474_0003_23676390: 3.96 MP
- europeana_00674544: 3.59 MP
- europeana_00675329: 3.83 MP
- europeana_00675495: 4.14 MP

## Verdict guide (COLUMN_COUNT_METHOD.md §3.4)

- Median ≈ 3 MP or above on the dense fixtures: optics were sufficient — the undercount
  was procedural; the enumeration protocol carries the fix.
- Median ≈ 1 MP or below: optics were binding — raise the serving-side pixel budget
  (config.VLLM_MM_MAX_PIXELS) before crediting or blaming any prompt.