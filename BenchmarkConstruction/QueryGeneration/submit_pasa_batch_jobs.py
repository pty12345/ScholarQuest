from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "QueryGeneration" / "output" / "pasa_batch_inputs"
DEFAULT_RESULT_DIR = ROOT / "QueryGeneration" / "output" / "pasa_batch_results"
DEFAULT_MANIFEST = DEFAULT_INPUT_DIR / "submitted_batches.json"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit, inspect, and download DashScope Batch API jobs.")
    parser.add_argument(
        "command",
        choices=("submit", "status", "download"),
        help="Action to run.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing batch input JSONL files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the submitted batch manifest.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory for downloaded batch output JSONL files.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the DashScope API key.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible DashScope base URL.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        choices=("24h",),
        help="Batch completion window.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of input JSONL files to submit.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="One-based index of the first sorted input JSONL file to submit.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="One-based index of the last sorted input JSONL file to submit.",
    )
    return parser.parse_args()


def _client(api_key_env: str, base_url: str) -> OpenAI:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key from environment variable {api_key_env}.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in manifest: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")


def submit_batches(
    client: OpenAI,
    *,
    input_dir: Path,
    manifest_path: Path,
    completion_window: str,
    limit: int | None,
    start_index: int,
    end_index: int | None,
) -> None:
    input_paths = sorted(input_dir.glob("pasa_qwen_batch_input_*.jsonl"))
    if not input_paths:
        raise FileNotFoundError(f"No batch input files found in {input_dir}")
    if start_index <= 0:
        raise ValueError("--start-index must be positive.")
    if end_index is not None and end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index.")
    input_paths = input_paths[start_index - 1 : end_index]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        input_paths = input_paths[:limit]

    manifest = _read_manifest(manifest_path) if manifest_path.exists() else []
    submitted_paths = {str(row.get("input_path", "")) for row in manifest}
    for input_path in input_paths:
        if str(input_path) in submitted_paths:
            print(f"Skip already submitted file: {input_path.name}")
            continue
        uploaded_file = client.files.create(file=input_path, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window=cast(Literal["24h"], completion_window),
            metadata={"ds_name": input_path.stem},
        )
        row = {
            "input_path": str(input_path),
            "file_id": uploaded_file.id,
            "batch_id": batch.id,
            "status": batch.status,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
        }
        manifest.append(row)
        print(f"Submitted {input_path.name}: file_id={uploaded_file.id}, batch_id={batch.id}, status={batch.status}")

    _write_manifest(manifest_path, manifest)
    print(f"Saved manifest: {manifest_path}")


def refresh_status(client: OpenAI, *, manifest_path: Path) -> None:
    manifest = _read_manifest(manifest_path)
    refreshed: list[dict[str, Any]] = []
    for row in manifest:
        batch_id = str(row["batch_id"])
        batch = client.batches.retrieve(batch_id)
        updated = dict(row)
        updated["status"] = batch.status
        updated["output_file_id"] = getattr(batch, "output_file_id", None)
        updated["error_file_id"] = getattr(batch, "error_file_id", None)
        refreshed.append(updated)
        print(
            f"{batch_id}: status={batch.status}, "
            f"output_file_id={updated['output_file_id']}, error_file_id={updated['error_file_id']}"
        )
    _write_manifest(manifest_path, refreshed)
    print(f"Updated manifest: {manifest_path}")


def download_results(client: OpenAI, *, manifest_path: Path, result_dir: Path) -> None:
    manifest = _read_manifest(manifest_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    for row in manifest:
        batch_id = str(row["batch_id"])
        output_file_id = row.get("output_file_id")
        if not isinstance(output_file_id, str) or not output_file_id:
            print(f"Skip {batch_id}: output_file_id is not ready.")
            continue

        content = client.files.content(output_file_id)
        output_path = result_dir / f"{batch_id}_output.jsonl"
        output_path.write_bytes(content.read())
        downloaded_count += 1
        print(f"Downloaded {batch_id}: {output_path}")

        error_file_id = row.get("error_file_id")
        if isinstance(error_file_id, str) and error_file_id:
            error_content = client.files.content(error_file_id)
            error_path = result_dir / f"{batch_id}_error.jsonl"
            error_path.write_bytes(error_content.read())
            print(f"Downloaded {batch_id} errors: {error_path}")

    print(f"Downloaded files: {downloaded_count}")


def main() -> None:
    args = parse_args()
    client = _client(args.api_key_env, args.base_url)

    if args.command == "submit":
        submit_batches(
            client,
            input_dir=args.input_dir.resolve(),
            manifest_path=args.manifest.resolve(),
            completion_window=args.completion_window,
            limit=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
        )
    elif args.command == "status":
        refresh_status(client, manifest_path=args.manifest.resolve())
    elif args.command == "download":
        download_results(
            client,
            manifest_path=args.manifest.resolve(),
            result_dir=args.result_dir.resolve(),
        )


if __name__ == "__main__":
    main()
