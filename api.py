from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Lecture Search API")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("lecture1_30s")

class SearchResult(BaseModel):
    start: float
    end: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

@app.get("/search", response_model=SearchResponse)
def search(q: str):
    query_embedding = model.encode([q]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    search_results = [
        SearchResult(start=meta["start"], end=meta["end"], text=doc)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    return SearchResponse(query=q, results=search_results)