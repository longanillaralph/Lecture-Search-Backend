"""Build per-lecture Chroma indexes from generated chunk files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import YouTubeSource, get_chroma_client, index_lecture_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and index lecture chunks in ChromaDB.")
    parser.add_argument(
        "--lecture-id",
        required=True,
        help="Stable ID for this lecture (use the YouTube video ID for YouTube sources)",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("."),
        help="Directory containing chunks_<window>s.json files",
    )
    parser.add_argument(
        "--source-url",
        default="",
        help="Canonical source URL to store with each result",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Lecture title to store with each result",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[30, 60, 120],
        help="Chunk windows to index (default: 30 60 120)",
    )
    args = parser.parse_args()

    client = get_chroma_client()
    source = YouTubeSource(args.lecture_id, args.source_url)
    for window in args.windows:
        chunks_path = args.chunks_dir / f"chunks_{window}s.json"
        with chunks_path.open(encoding="utf-8") as chunks_file:
            chunks = json.load(chunks_file)
        result = index_lecture_chunks(
            client,
            lecture_id=args.lecture_id,
            source=source,
            title=args.title,
            chunks=chunks,
            window_seconds=window,
        )
        state = "already indexed" if result.already_indexed else "indexed"
        print(f"{result.collection_name}: {state} {result.chunk_count} chunks")


if __name__ == "__main__":
    main()
