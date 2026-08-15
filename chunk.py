"""Chunk a transcript JSON file for offline pipeline use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import chunk_by_time, load_transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk a cleaned transcript by elapsed time.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("transcript.json"),
        help="Transcript JSON input path (default: transcript.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for generated chunks_<window>s.json files",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[30, 60, 120],
        help="Chunk windows in seconds (default: 30 60 120)",
    )
    args = parser.parse_args()

    segments = load_transcript(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for window in args.windows:
        chunks = chunk_by_time(segments, window_seconds=window)
        output_path = args.output_dir / f"chunks_{window}s.json"
        with output_path.open("w", encoding="utf-8") as chunks_file:
            json.dump(chunks, chunks_file, indent=2, ensure_ascii=False)
        print(f"{output_path}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
