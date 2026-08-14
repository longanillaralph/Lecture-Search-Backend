import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("chunks.json") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db") #* gagawa ng persistent na database sa local storage 
collection = client.get_or_create_collection("lecture1")

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts).tolist() #* eto yung embeddings na gagamitin para sa indexing

#* Add the embeddings and metadata to the Chroma collection
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    embeddings=embeddings,
    documents=texts,
    metadatas=[{"start": c["start"], "end": c["end"]} for c in chunks]
)

print(f"Indexed {len(chunks)} chunks into Chroma")