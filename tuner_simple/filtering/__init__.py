"""Ownership-based line filtering, overlap merging, and final assignment.

The stable public entry point is
``line_filtering_v2_1_IoU_fast.filter_lines_for_alignment_by_ownership``.
Focused helper modules in this package own candidate coverage construction,
true-IoU overlap merging, geometry fitting, optional Cython helper imports, and
final prediction-column ownership.
"""
