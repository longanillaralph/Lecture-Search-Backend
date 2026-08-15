# Lecture Search — Application Documentation

## 1. What the app does

Lecture Search turns a public YouTube lecture into a searchable workspace. A user submits a YouTube URL, the backend obtains the lecture captions, creates searchable semantic chunks, and lets the user ask natural-language questions. Answers include transcript excerpts and links that open YouTube at the relevant timestamp.

The system has two repositories:

| Repository | Responsibility |
|---|---|
| `lecture-search-frontend` | React/Vite user interface, processing state, polling, questions, answers, source links |
| `lecture-search-backend` | FastAPI API, caption retrieval, chunking, embeddings, ChromaDB search, LLM answers |

## 2. End-to-end flow

```text
User enters YouTube URL
        ↓
Frontend POST /lectures
        ↓
Backend returns 202 + lecture_id
        ↓
Background job fetches captions from Supadata
        ↓
Captions are cleaned and grouped into time windows
        ↓
FastEmbed converts chunks into vectors
        ↓
ChromaDB stores the vectors for this lecture
        ↓
Frontend polls GET /lectures/{lecture_id}
        ↓
Workspace becomes ready
        ↓
User asks a question
        ↓
Semantic search returns relevant transcript chunks
        ↓
Optional NVIDIA/OpenAI-compatible LLM creates a grounded answer
```

The application does not download YouTube audio in the production API. This avoids media-download failures and reduces server resource usage.

## 3. Frontend behavior

The frontend starts on a landing page with a YouTube URL form. After the user submits a URL:

1. It validates the URL and supported YouTube host.
2. It calls `POST /lectures`.
3. It displays a processing state while polling the lecture status.
4. It opens the lecture workspace when the backend returns `ready`.
5. It sends questions to the current lecture ID only.
6. It renders the answer and timestamped source cards.

The interface is deployed on Vercel. Its only production secret/configuration is the public backend origin:

```text
VITE_API_URL=https://your-backend-domain.example.com
```

Provider keys must never be placed in the frontend.

## 4. Backend components

### `api.py`

FastAPI entrypoint. It provides:

- `GET /healthz` — service health check
- `POST /lectures` — starts asynchronous lecture processing
- `GET /lectures/{lecture_id}` — returns processing/index status
- `POST /lectures/{lecture_id}/search` — searches one indexed lecture
- `POST /search` — compatibility route for searching by URL or lecture ID

### `pipeline.py`

Core processing logic:

- Validates and normalizes YouTube URLs.
- Calls Supadata for captions.
- Cleans caption text and timestamps.
- Groups captions into configurable time windows.
- Creates embeddings with FastEmbed.
- Stores vectors and metadata in ChromaDB.
- Performs semantic search.
- Sends retrieved excerpts to an OpenAI-compatible chat-completions API.

### `requirements.txt`

Production dependencies only. Whisper is not needed by the deployed caption workflow. The optional `requirements-offline.txt` adds `faster-whisper` for local audio-file experiments.

## 5. Data model

### Lecture identity

The YouTube video ID is the stable `lecture_id`. For example:

```text
https://youtu.be/VIDEO_ID
lecture_id = VIDEO_ID
```

### Chroma collection

Each lecture is stored in a collection named approximately:

```text
lecture_<video_id>_<window_seconds>s
```

Each vector contains:

- `document` — transcript text
- `start` — start time in seconds
- `end` — end time in seconds
- `source_url` — canonical YouTube URL
- `video_id` — YouTube ID
- `lecture_id` — application lecture ID
- `title` — current lecture title value

### Search response

```json
{
  "query": "What is a limit?",
  "lecture_id": "VIDEO_ID",
  "answer": "...",
  "results": [
    {
      "source_url": "https://www.youtube.com/watch?v=VIDEO_ID",
      "start": 532.0,
      "end": 592.0,
      "text": "..."
    }
  ]
}
```

## 6. Environment configuration

Caption provider:

```text
SUPADATA_API_KEY=...
```

LLM provider example using NVIDIA NIM:

```text
LLM_API_KEY=...
LLM_MODEL=meta/llama-3.1-8b-instruct
LLM_CHAT_COMPLETIONS_URL=https://integrate.api.nvidia.com/v1/chat/completions
```

Storage and performance:

```text
CHROMA_DB_PATH=/data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=8
CHUNK_WINDOW_SECONDS=60
MAX_TRANSCRIPT_SEGMENTS=3000
```

Browser access:

```text
CORS_ORIGINS=https://lecture-search-frontend.vercel.app
```

## 7. Deployment

### Backend

The backend can run on Render, Railway, or another Python host:

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port $PORT
```

The host must:

- Provide Python 3.12.
- Expose the service on `0.0.0.0:$PORT`.
- Provide persistent storage for `/data/chroma_db`.
- Provide enough memory for ChromaDB and FastEmbed.
- Store Supadata and LLM keys as backend-only variables.

### Frontend

Vercel builds the Vite app from the frontend repository. Set `VITE_API_URL` to the backend public origin, then redeploy because Vite embeds environment variables during the build.

## 8. Common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `503 hibernate-wake-error` | Free backend is waking | Retry or use a non-sleeping instance |
| `502` during processing | Backend crash, timeout, or provider error | Check backend logs and memory |
| Captions unavailable | Video is restricted or provider cannot access it | Try a public video with accessible captions |
| Lecture ready but answer fails | LLM key, model, or endpoint is wrong | Verify all three LLM variables |
| Browser CORS error | Frontend origin is not allowed | Add the exact Vercel origin to `CORS_ORIGINS` |
| Data disappears after redeploy | Chroma path is ephemeral | Attach persistent storage and use `/data/chroma_db` |

## 9. Security rules

- Never commit `.env` files or provider keys.
- Rotate any key shown in a screenshot, terminal, commit, or chat.
- Keep Supadata and LLM calls on the backend.
- Restrict CORS to known frontend origins.
- Add authentication and rate limiting before opening the app to the public.

## 10. Current limitations

- Processing state is currently held in backend memory and can be lost if the service restarts.
- ChromaDB is local to the backend service and requires persistent storage.
- Caption availability depends on the external transcript provider.
- There is no user account, lecture ownership, deletion, or usage quota system yet.
- The default lecture title is based on the video ID rather than fetched YouTube metadata.
