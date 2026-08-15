"""Search one lecture from the command line."""

from __future__ import annotations

import argparse

from pipeline import get_chroma_client, search_lecture


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about one indexed lecture.")
    parser.add_argument(
        "--lecture-id",
        required=True,
        help="The lecture_id returned by the ingestion/API pipeline",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("question", nargs="?", help="Question; prompts when omitted")
    args = parser.parse_args()

    question = args.question or input("Ask a question about the lecture: ")
    results = search_lecture(
        get_chroma_client(), args.lecture_id, question, top_k=args.top_k
    )
    for result in results:
        print(f"\n[{result['start']:.0f}s -> {result['end']:.0f}s]")
        print(result["text"])


if __name__ == "__main__":
    main()
