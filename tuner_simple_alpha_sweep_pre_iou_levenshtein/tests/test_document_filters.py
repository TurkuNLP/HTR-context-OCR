from __future__ import annotations

from tuner_simple_alpha_sweep_pre_iou_levenshtein.document_selection.document_filters import select_documents
from tuner_simple_alpha_sweep_pre_iou_levenshtein.document_selection.runfile_loader import RunfileDocument


# Define the make_document function; its body below performs one named step of the pipeline.
def make_document(index: int, fname: str, language: str, document_type: str) -> RunfileDocument:
    # Return this computed value to the caller so the next pipeline stage can use it.
    return RunfileDocument(
        # Pass document_index into the surrounding call; this supplies the document position from the runfile, preserved for auditability.
        document_index=index,
        # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
        fname=fname,
        # Pass main_language into the surrounding call; this supplies the document language label used for filtering and stitched plot grouping.
        main_language=language,
        # Pass document_type into the surrounding call; this supplies the document type label used for filtering and summary grouping.
        document_type=document_type,
        # Pass reference_text into the surrounding call; this supplies the normalized reference transcription for this document.
        reference_text="reference text long enough for a window",
        # Pass prediction_text into the surrounding call; this supplies the normalized model prediction for this document.
        prediction_text="prediction text long enough for a window",
    )


# Define the test_select_documents_filters_language_type_name_and_max_items function; its body below performs one named step of the pipeline.
def test_select_documents_filters_language_type_name_and_max_items() -> None:
    # Compute or store documents so later code can reuse this named value clearly.
    documents = [
        # Pass this value into the surrounding multi-line call or collection.
        make_document(0, "a.jpeg", "Finnish", "print"),
        # Pass this value into the surrounding multi-line call or collection.
        make_document(1, "b.jpeg", "Finnish", "handwriting"),
        # Pass this value into the surrounding multi-line call or collection.
        make_document(2, "c.jpeg", "German", "print"),
    ]
    # Compute or store messages: list[str] so later code can reuse this named value clearly.
    messages: list[str] = []

    # Compute or store selected so later code can reuse this named value clearly.
    selected = select_documents(
        # Pass the documents argument into the surrounding call so the callee receives that setting explicitly.
        documents=documents,
        # Pass languages into the surrounding call; this supplies the optional language filter requested by the user.
        languages=("Finnish",),
        # Pass document_types into the surrounding call; this supplies the optional document-type filter requested by the user.
        document_types=("print",),
        # Pass target_fnames into the surrounding call; this supplies the optional exact filename filter requested by the user.
        target_fnames=("a.jpeg",),
        # Pass max_items into the surrounding call; this supplies the optional cap on how many selected documents are processed.
        max_items=1,
        # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
        log=messages.append,
    )

    # Verify this expected condition during tests so regressions fail clearly.
    assert [document.fname for document in selected] == ["a.jpeg"]
    # Verify this expected condition during tests so regressions fail clearly.
    assert messages[-1].endswith("1 documents")
