"""Transcribe a supplied local lecture recording."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import save_transcript, transcribe_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a lecture recording with Whisper.")
    parser.add_argument("input_file", type=Path, help="Path to the downloaded audio/video file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("transcript.json"),
        help="Transcript JSON output path (default: transcript.json)",
    )
    args = parser.parse_args()

    segments = transcribe_audio(args.input_file)
    save_transcript(segments, args.output)
    print(f"Saved {len(segments)} cleaned segments to {args.output}")


if __name__ == "__main__":
    main()
