from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from document_selection.document_filters import select_documents
    from document_selection.runfile_loader import load_runfile_documents
    from dynamic_pool.document_pool import DocumentPool, document_filename, write_json_file_atomically
else:
    from ..document_selection.document_filters import select_documents
    from ..document_selection.runfile_loader import load_runfile_documents
    from .document_pool import DocumentPool, document_filename, write_json_file_atomically


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or resume a tuner_simple dynamic document pool.")
    parser.add_argument("--runfile-json", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--document-type", action="append", default=[])
    parser.add_argument("--all-languages", action="store_true")
    parser.add_argument("--all-document-types", action="store_true")
    parser.add_argument("--target-fname", action="append", default=[])
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--window-stride", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--requeue-claimed", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def quiet_log(message: str) -> None:
    return None


def pool_has_existing_documents(pool_dir: Path) -> bool:
    state_directories = ("available", "claimed", "done", "failed")
    for state_directory_name in state_directories:
        state_directory_path = Path(pool_dir) / state_directory_name
        if any(state_directory_path.glob("document_*.json")):
            return True
    if (Path(pool_dir) / "selected_documents.jsonl").exists():
        return True
    return False


def initialize_pool(arguments: argparse.Namespace) -> dict[str, int]:
    document_pool = DocumentPool(arguments.pool_dir)
    document_pool.ensure_state_directories()
    if arguments.resume:
        if arguments.requeue_claimed:
            document_pool.requeue_claimed_documents()
        if arguments.retry_failed:
            document_pool.requeue_failed_documents()
        return document_pool.state_counts()
    if pool_has_existing_documents(arguments.pool_dir):
        raise SystemExit(f"Pool already exists at {arguments.pool_dir}. Use --resume or choose a fresh --output-dir.")
    runfile_documents = load_runfile_documents(arguments.runfile_json)
    selected_documents = select_documents(
        documents=runfile_documents,
        languages=tuple() if arguments.all_languages else tuple(arguments.language or []),
        document_types=tuple() if arguments.all_document_types else tuple(arguments.document_type or []),
        target_fnames=tuple(arguments.target_fname or []),
        max_items=arguments.max_items,
        log=quiet_log,
    )
    manifest_path = arguments.pool_dir / "selected_documents.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for pool_ordinal, document in enumerate(selected_documents):
            payload = {
                "pool_ordinal": int(pool_ordinal),
                "document_index": int(document.document_index),
                "filename": str(document.fname),
                "main_language": str(document.main_language),
                "document_type": str(document.document_type),
                "window_size": int(arguments.window_size),
                "window_stride": int(arguments.window_stride),
            }
            write_json_file_atomically(arguments.pool_dir / "available" / document_filename(pool_ordinal), payload)
            manifest_handle.write(json.dumps(payload, sort_keys=True) + "\n")
    document_pool.write_event("initialized", {"document_count": len(selected_documents)})
    return document_pool.state_counts()


def main() -> int:
    arguments = parse_arguments()
    state_counts = initialize_pool(arguments)
    print("dynamic document pool state:")
    for state_name in sorted(state_counts):
        print(f"  {state_name}: {state_counts[state_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
