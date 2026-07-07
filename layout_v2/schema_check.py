"""Small JSON-schema subset validator for the layout_v2 pass contracts.

The live server should enforce these schemas, but the runner must not trust that blindly.  This
module validates only the schema features used by our pass definitions, keeping the boundary
check auditable and dependency-free on the compute nodes.
"""

from __future__ import annotations

from typing import Any


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Return human-readable schema errors for the JSON-schema subset layout_v2 uses."""
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        errors.append(f"{path}: expected {_type_label(expected_type)}, got {_json_type(value)}")
        return errors

    if isinstance(value, dict):
        errors.extend(_validate_object(value, schema, path))
    elif isinstance(value, list):
        errors.extend(_validate_array(value, schema, path))
    else:
        errors.extend(_validate_scalar(value, schema, path))
    return errors


def short_error(errors: list[str], limit: int = 4) -> str:
    """Compact an error list for per-pass records and retry logs."""
    shown = errors[:limit]
    suffix = f"; ... +{len(errors) - limit} more" if len(errors) > limit else ""
    return "; ".join(shown) + suffix


def _validate_object(value: dict, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    for key in required:
        if key not in value:
            errors.append(f"{path}: missing required key {key}")

    if schema.get("additionalProperties") is False:
        for key in value:
            if key not in properties:
                errors.append(f"{path}.{key}: additional property not allowed")

    for key, child_schema in properties.items():
        if key in value:
            errors.extend(validate(value[key], child_schema, f"{path}.{key}"))
    return errors


def _validate_array(value: list, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append(f"{path}: expected at least {min_items} items, got {len(value)}")
    if isinstance(max_items, int) and len(value) > max_items:
        errors.append(f"{path}: expected at most {max_items} items, got {len(value)}")

    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            errors.extend(validate(item, item_schema, f"{path}[{index}]"))
    return errors


def _validate_scalar(value: Any, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(v) for v in schema["enum"])
        errors.append(f"{path}: expected one of {allowed}, got {value!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: expected >= {minimum}, got {value}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: expected <= {maximum}, got {value}")

    max_length = schema.get("maxLength")
    if isinstance(value, str) and isinstance(max_length, int) and len(value) > max_length:
        errors.append(f"{path}: expected length <= {max_length}, got {len(value)}")
    return errors


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(value, one) for one in expected)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _type_label(expected: Any) -> str:
    if isinstance(expected, list):
        return " or ".join(str(item) for item in expected)
    return str(expected)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__
