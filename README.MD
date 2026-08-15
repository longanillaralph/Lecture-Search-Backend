# Lecture Search — Backend

FastAPI backend for turning public YouTube captions into a searchable lecture workspace. Users can ask questions about a lecture and receive grounded answers with timestamped transcript sources.

The frontend lives in the separate [lecture-search-frontend](https://github.com/longanillaralph/lecture-search-frontend) repository.

## Architecture

```text
YouTube URL → Supadata captions → timestamped chunks → FastEmbed → ChromaDB → semantic search + optional LLM answer
```

The API does not download YouTube audio. It retrieves captions through Supadata, avoiding server-side YouTube media restrictions.

## API workflow

`POST /lectures` starts asynchronous processing and returns `202 Accepted` with a lecture ID. Poll `GET /lectures/{lecture_id}` until the status becomes `ready`.

```json
{ "url": "https://www.youtube.com/watch?v=VIDEO_ID" }
```

Then search the completed lecture:

```http
POST /lectures/VIDEO_ID/search
Content-Type: application/json
```

```json
{ "q": "What is a limit?", "generate_answer": true, "top_k": 3 }
```

`GET /healthz` is the hosting health check. Interactive API documentation is available at `/docs`.

## Environment variables

```text
SUPADATA_API_KEY=...
LLM_API_KEY=...
LLM_MODEL=meta/llama-3.1-8b-instruct
LLM_CHAT_COMPLETIONS_URL=https://integrate.api.nvidia.com/v1/chat/completions
CORS_ORIGINS=https://lecture-search-frontend.vercel.app
CHROMA_DB_PATH=/data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=8
CHUNK_WINDOW_SECONDS=60
MAX_TRANSCRIPT_SEGMENTS=3000
```

Never commit or expose API keys. Store them only in the backend hosting provider.

## Deployment

The service requires Python 3.12 and must listen on the host-provided `PORT`:

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port $PORT
```

For Render or Railway, connect this GitHub repository, add the environment variables, and attach persistent storage mounted at `/data` for ChromaDB. Use one worker because Chroma writes locally.

## Local development

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uvicorn api:app --reload
```

The API runs at `http://localhost:8000`; docs are at `http://localhost:8000/docs`.

The optional offline Whisper workflow is separated into `requirements-offline.txt` and is not required by the deployed caption-based API.

## Stack

| Layer | Tool |
|---|---|
| API | FastAPI + Uvicorn |
| Captions | Supadata |
| Embeddings | FastEmbed |
| Vector store | ChromaDB |
| Answer generation | OpenAI-compatible chat-completions API |
| Runtime | Python 3.12 |
