import json
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

with open("test_questions.json") as f:
    test_questions = json.load(f)

def is_correct(results, correct_start, tolerance=30):
    for meta in results["metadatas"][0]:
        if abs(meta["start"] - correct_start) <= tolerance:
            return True
    return False

scores = {}

for window in [30, 60, 120]:
    collection_name = f"lecture1_{window}s"
    collection = client.get_collection(collection_name)

    correct_count = 0
    for item in test_questions:
        query_embedding = model.encode([item["question"]]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)

        if is_correct(results, item["correct_start"]):
            correct_count += 1
        else:
            print(f"  [{window}s MISSED] \"{item['question']}\" (expected ~{item['correct_start']}s)")

    scores[window] = correct_count
    print(f"\n{window}s window: {correct_count}/{len(test_questions)} correct\n")

print("=== Summary ===")
for window, score in scores.items():
    print(f"{window}s: {score}/{len(test_questions)}")