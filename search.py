import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db") #* bubuksan niya yung persistent na database sa local storage
collection = client.get_collection("lecture1") #* bubuksan niya yung collection na lecture1 sa loob ng database

query = input("Ask a question about the lecture: ") #* Eto yung input ng user na query
query_embedding = model.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=3) #* Eto yung magreresult ng top 3 na pinaka-malapit na chunks sa query ng user

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n[{meta['start']:.0f}s -> {meta['end']:.0f}s]")
    print(doc)