"""layout_v2 passes: one module per model call (prompt + schema + message builder + postprocess).

Each pass module exposes the same four names so the runner can treat them uniformly:
- ``SYSTEM_PROMPT``  the system message (task instructions only — no per-document content);
- ``SCHEMA``         the guided-JSON schema (property order = emission order: evidence first);
- ``build_user_parts(...)``  the per-document user turn (text preamble + images);
- ``postprocess(parsed)``    light normalization of the parsed answer (clamping, sorting).
"""
