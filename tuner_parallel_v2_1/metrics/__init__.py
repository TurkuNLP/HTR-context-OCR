"""Metric helpers for ``tuner_parallel_v2_1``.

The tuner keeps local scoring helpers and the required v2.12-compatible metric
semantics in this package.  ``v2_12_metric_adapter.py`` is the public boundary:
normal runs use the local compatibility code, while the historical external
``text_metrics_v2_12_parallel`` tree is loaded only for audits and equivalence
tests.
"""
