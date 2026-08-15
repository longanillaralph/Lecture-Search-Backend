import json
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

for window in [30, 60, 120]:
    filename = f"chunks_{window}s.json"
    with open(filename) as f:
        chunks = json.load(f)

    collection_name = f"lecture1_{window}s"
    collection = client.get_or_create_collection(collection_name)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"start": c["start"], "end": c["end"]} for c in chunks]
    )

    print(f"{collection_name}: indexed {len(chunks)} chunks")