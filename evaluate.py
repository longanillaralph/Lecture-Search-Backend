"""Evaluate retrieval quality for one lecture's configured chunk windows."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pipeline import collection_name_for, embed_texts, get_chroma_client


def is_correct(results: Any, correct_start: float, tolerance: float = 30) -> bool:
    metadatas = results.get("metadatas") or []
    if not metadatas:
        return False
    for metadata in metadatas[0]:
        if abs(float(metadata["start"]) - correct_start) <= tolerance:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval for one indexed lecture.")
    parser.add_argument("--lecture-id", required=True)
    parser.add_argument("--questions", default="test_questions.json")
    parser.add_argument("--windows", type=int, nargs="+", default=[30, 60, 120])
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as questions_file:
        test_questions = json.load(questions_file)

    client = get_chroma_client()
    scores: dict[int, int] = {}
    for window in args.windows:
        collection = client.get_collection(collection_name_for(args.lecture_id, window))
        correct_count = 0
        for item in test_questions:
            query_embedding = embed_texts([item["question"]])[0]
            results = collection.query(query_embeddings=[query_embedding], n_results=3)

            if is_correct(results, item["correct_start"]):
                correct_count += 1
            else:
                print(
                    f"  [{window}s MISSED] \"{item['question']}\" "
                    f"(expected ~{item['correct_start']}s)"
                )

        scores[window] = correct_count
        print(f"\n{window}s window: {correct_count}/{len(test_questions)} correct\n")

    print("=== Summary ===")
    for window, score in scores.items():
        print(f"{window}s: {score}/{len(test_questions)}")


if __name__ == "__main__":
    main()
