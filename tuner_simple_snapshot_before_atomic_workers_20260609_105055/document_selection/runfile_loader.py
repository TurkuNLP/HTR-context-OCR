from __future__ import annotations

"""Load normalized document records from a Churro outputs.json runfile."""

from dataclasses import dataclass
import json
from pathlib import Path


# Ask Python to generate common data-container methods for the class defined next.
@dataclass(frozen=True)
# Define the RunfileDocument class, which groups related state and behavior for this part of the pipeline.
class RunfileDocument:
    """One document entry selected from the runfile."""

    # Define the document_index field; it stores the document position from the runfile, preserved for auditability.
    document_index: int
    # Define the fname field; it stores the document filename used to match runfile records to score matrices.
    fname: str
    # Define the main_language field; it stores the document language label used for filtering and stitched plot grouping.
    main_language: str
    # Define the document_type field; it stores the document type label used for filtering and summary grouping.
    document_type: str
    # Define the reference_text field; it stores the normalized reference transcription for this document.
    reference_text: str
    # Define the prediction_text field; it stores the normalized model prediction for this document.
    prediction_text: str


# Define the load_runfile_documents function; its body below performs one named step of the pipeline.
def load_runfile_documents(runfile_json: Path) -> list[RunfileDocument]:
    """Return normalized runfile records in their original JSON order."""
    # Compute or store payload so later code can reuse this named value clearly.
    payload = json.loads(Path(runfile_json).read_text(encoding="utf-8"))
    # Check whether not isinstance(payload, list); the indented block handles that specific case.
    if not isinstance(payload, list):
        # Stop execution for this invalid state by raising an explicit exception.
        raise ValueError(f"Expected runfile JSON list, got {type(payload).__name__}")

    # Compute or store documents: list[RunfileDocument] so later code can reuse this named value clearly.
    documents: list[RunfileDocument] = []
    # Iterate over document_index, item in enumerate(payload) so each item is processed with the same logic.
    for document_index, item in enumerate(payload):
        # Check whether not isinstance(item, dict); the indented block handles that specific case.
        if not isinstance(item, dict):
            # Skip the rest of this loop iteration and move to the next item.
            continue
        # Compute or store file_name so later code can reuse this named value clearly.
        file_name = item.get("file_name", item.get("fname", f"document_{document_index:06d}"))
        # Add this item to the list that is accumulating results for later output.
        documents.append(
            # Start a multi-line call or data structure so related arguments stay readable.
            RunfileDocument(
                # Pass document_index into the surrounding call; this supplies the document position from the runfile, preserved for auditability.
                document_index=int(document_index),
                # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
                fname=Path(str(file_name)).name,
                # Pass main_language into the surrounding call; this supplies the document language label used for filtering and stitched plot grouping.
                main_language=str(item.get("main_language", "")),
                # Pass document_type into the surrounding call; this supplies the document type label used for filtering and summary grouping.
                document_type=str(item.get("document_type", "")),
                # Pass reference_text into the surrounding call; this supplies the normalized reference transcription for this document.
                reference_text=str(item.get("normalized_gold_text", item.get("ref", ""))),
                # Pass prediction_text into the surrounding call; this supplies the normalized model prediction for this document.
                prediction_text=str(item.get("normalized_predicted_text", item.get("pred", ""))),
            )
        )
    # Return this computed value to the caller so the next pipeline stage can use it.
    return documents


__all__ = ["RunfileDocument", "load_runfile_documents"]
