from __future__ import annotations

"""Small, explicit filters for runfile document selection."""

from collections.abc import Callable
from pathlib import Path

from .runfile_loader import RunfileDocument

# Compute or store LogFn so later code can reuse this named value clearly.
LogFn = Callable[[str], None]


# Define the _log_count function; its body below performs one named step of the pipeline.
def _log_count(log: LogFn, label: str, documents: list[RunfileDocument]) -> None:
    """Emit one selection count with a stable label."""
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[selection] {label}: {len(documents)} documents")


# Define the select_documents function; its body below performs one named step of the pipeline.
def select_documents(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the documents field so this data object records that value explicitly.
    documents: list[RunfileDocument],
    # Define the languages field; it stores the optional language filter requested by the user.
    languages: tuple[str, ...],
    # Define the document_types field; it stores the optional document-type filter requested by the user.
    document_types: tuple[str, ...],
    # Define the target_fnames field; it stores the optional exact filename filter requested by the user.
    target_fnames: tuple[str, ...],
    # Define the max_items field; it stores the optional cap on how many selected documents are processed.
    max_items: int | None,
    # Define the log field so this data object records that value explicitly.
    log: LogFn,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> list[RunfileDocument]:
    """Apply user-requested filters in a predictable order."""
    # Compute or store selected so later code can reuse this named value clearly.
    selected = list(documents)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    _log_count(log, "loaded from runfile", selected)

    # Check whether languages; the indented block handles that specific case.
    if languages:
        # Compute or store wanted_languages so later code can reuse this named value clearly.
        wanted_languages = {str(value) for value in languages}
        # Compute or store selected so later code can reuse this named value clearly.
        selected = [document for document in selected if document.main_language in wanted_languages]
        # Execute this statement as the next small step in the surrounding pipeline logic.
        _log_count(log, f"after language filter {sorted(wanted_languages)!r}", selected)

    # Check whether document_types; the indented block handles that specific case.
    if document_types:
        # Compute or store wanted_document_types so later code can reuse this named value clearly.
        wanted_document_types = {str(value) for value in document_types}
        # Compute or store selected so later code can reuse this named value clearly.
        selected = [document for document in selected if document.document_type in wanted_document_types]
        # Execute this statement as the next small step in the surrounding pipeline logic.
        _log_count(log, f"after document-type filter {sorted(wanted_document_types)!r}", selected)

    # Check whether target_fnames; the indented block handles that specific case.
    if target_fnames:
        # Compute or store wanted_names so later code can reuse this named value clearly.
        wanted_names = {Path(str(value)).name for value in target_fnames}
        # Compute or store selected so later code can reuse this named value clearly.
        selected = [document for document in selected if Path(document.fname).name in wanted_names]
        # Execute this statement as the next small step in the surrounding pipeline logic.
        _log_count(log, f"after target filename filter {sorted(wanted_names)!r}", selected)

    # Check whether max_items is not None; the indented block handles that specific case.
    if max_items is not None:
        # Compute or store selected so later code can reuse this named value clearly.
        selected = selected[: int(max_items)]
        # Execute this statement as the next small step in the surrounding pipeline logic.
        _log_count(log, f"after max-items={int(max_items)}", selected)

    # Return this computed value to the caller so the next pipeline stage can use it.
    return selected


__all__ = ["select_documents"]
