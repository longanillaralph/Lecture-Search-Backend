from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lecture Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lecture-search-frontend.vercel.app"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
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
    query_embedding = list(model.embed([q]))[0].tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    search_results = [
        SearchResult(start=meta["start"], end=meta["end"], text=doc)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    return SearchResponse(query=q, results=search_results)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)