# Lecture Search Tool — Backend

Upload a lecture recording, get back a searchable transcript. Ask a question about the lecture in plain language and jump straight to the timestamp where it was discussed — instead of scrubbing through a 75-minute recording by hand.

This is the backend: transcription, chunking, embedding, search, and evaluation. It exposes both a console interface and a FastAPI HTTP API. For the React frontend that talks to this API, see [lecture-search-frontend](https://github.com/longanillaralph/lecture-search-frontend).

## Deployment

Use Python 3.12, install `requirements.txt`, and start the web service with:

```bash
bash render-build.sh
bash render-start.sh
```

On Render, set the build command to `bash render-build.sh` and the start command to `bash render-start.sh`. These scripts install Deno and expose it to `yt-dlp`; current YouTube extraction requires a JavaScript runtime and the `yt-dlp[default]` EJS dependency.

Configure these environment variables in the hosting provider:

- `LLM_API_KEY` (or `OPENAI_API_KEY`) for generated answers
- `LLM_MODEL` and, for OpenAI-compatible providers, `LLM_CHAT_COMPLETIONS_URL`
- `CORS_ORIGINS` as a comma-separated list containing the exact deployed frontend origin
- `CHROMA_DB_PATH` pointing at persistent storage; the default local path is not durable on ephemeral hosts

Set the provider health check to `GET /healthz`. Keep this service at one worker unless the Chroma path is shared by a storage layer that supports concurrent writers. Lecture ingestion is synchronous and CPU-intensive, so the host must allow requests to run for several minutes.

## How it works

The HTTP API processes each submitted YouTube video independently. Its `lecture_id` is the YouTube video ID, so every search is scoped to the lecture selected by the user. The command-line scripts remain available for local/offline processing.

1. **Download audio** — grab a lecture recording from YouTube using `yt-dlp` (or use any local audio/video file).
2. **Transcription** — downloads the source to a temporary directory and runs it through Whisper (`faster-whisper`) to produce a cleaned, timestamped transcript.
3. **`chunk.py`** — groups the transcript into bigger windows. Generates three sizes to compare: `chunks_30s.json`, `chunks_60s.json`, `chunks_120s.json`.
4. **Indexing** — embeds each chunk with FastEmbed and stores it in a Chroma collection named for that lecture (for example, `lecture_<video_id>_30s`).
5. **`evaluate.py`** — runs labeled test questions against every chunk size and measures retrieval accuracy, producing the evidence behind the chunk-size decision (see below).
6. **Query the index** — `POST /lectures/{lecture_id}/search` performs semantic search only within that lecture and can generate a grounded answer with timestamped sources.

```
YouTube link + lecture_id
     |
     v
  yt-dlp  ---->  temporary audio file
                       |
                       v
                Whisper  ---->  cleaned transcript
                       |
                       v
                  chunking      ---->  30-second chunks
                       |
                       v
                  indexing      ---->  chroma_db/lecture_<video_id>_30s
                       |
                       v
       semantic search
              |
       answer + sources
```

## HTTP API

Process a lecture once, then use the returned ID for all questions:

```http
POST /lectures
Content-Type: application/json

{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}
```

```http
POST /lectures/VIDEO_ID/search
Content-Type: application/json

{"question":"What is a limit?","top_k":3,"generate_answer":true}
```

The response contains `answer` and `results`. Each result includes the source URL, transcript text, and `start`/`end` timestamps. Set `LLM_API_KEY` (or `OPENAI_API_KEY`) and optionally `LLM_MODEL`/`LLM_CHAT_COMPLETIONS_URL` to enable answer generation. Retrieval remains available with `generate_answer: false`.

## Why semantic search, not keyword search

The search doesn't look for exact word matches — it matches on *meaning*. Asking "what is limits?" correctly returns chunks discussing left-side/right-side limits and limit existence, even without the word "limits" necessarily appearing verbatim in every match. This works because embeddings place text with similar meaning close together in vector space, regardless of the exact words used.

## Chunk size evaluation

The "messy decision" this project is built around is chunk size — too small and a chunk loses the context that makes it meaningful; too large and it returns more audio than the user actually wanted. Rather than guessing, three window sizes were tested (30s, 60s, 120s) against 7 real questions about the lecture, each with a manually-verified correct timestamp.

**Results:**

| Window size | Score |
|---|---|
| 30s | 4/7 |
| 60s | 4/7 |
| 120s | 2/7 |

- 30s and 60s tied on accuracy; 120s clearly underperformed, consistent with the expected failure mode — larger chunks blend multiple topics together, diluting the embedding and making it less precise for narrow questions.
- 2 of 7 questions failed at every window size. The corresponding timestamps were manually re-verified as correct, ruling out chunking as the cause — this points instead to a limitation in the `tiny` Whisper model's transcription accuracy or the embedding model itself.
- **Selected: 30-second windows.** With 30s and 60s tied on accuracy, the tie-breaker was timestamp precision — a 30-second window lands a user closer to the exact moment they're looking for.

Full technical detail on every script and data format lives in [`DOCUMENTATION.md`](./DOCUMENTATION.md).

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Usage

```bash
# 1. Optional offline workflow: download a lecture (or supply a local file)
yt-dlp -x --audio-format mp3 "YOUTUBE_URL"

# 2. Transcribe the supplied file; there is no filename hardcoded in the script
python transcribe.py path/to/lecture.webm

# 3. Chunk the transcript into three window sizes
python chunk.py

# 4. Build per-lecture indexes
python index.py --lecture-id VIDEO_ID --source-url "https://www.youtube.com/watch?v=VIDEO_ID"

# 5. Evaluate which chunk size performs best
python evaluate.py

# 6a. Search one lecture from the console
python search.py --lecture-id VIDEO_ID

# 6b. OR run the API (needed if using the React frontend)
uvicorn api:app --reload
# Interactive docs: http://localhost:8000/docs
```

## Current status

- [x] Transcription pipeline working (tested on a 75-minute calculus lecture, 1618 raw segments)
- [x] Chunking into three window sizes (30s: 157 chunks, 60s: 82 chunks, 120s: 43 chunks)
- [x] Embedding + Chroma indexing (three separate collections)
- [x] Semantic search returning relevant, correctly-distinguished results across different topics in the same lecture
- [x] Compared chunk window sizes against 7 labeled test questions — 30-second windows selected
- [x] FastAPI HTTP endpoint with CORS support
- [x] React frontend (separate repo) with styled UI and error handling
- [x] Dynamic processing and per-lecture collections for supported YouTube URLs
- [x] HTTP search scoped by `lecture_id`, with optional grounded LLM answers
- [ ] Deploy backend + frontend

## Resume bullet

Built a lecture search tool that transcribes audio and enables semantic search over timestamped segments; evaluated multiple chunking strategies against retrieval quality and selected 30-second windows, chosen for retrieval accuracy on par with larger windows while returning more precise timestamps.

## Stack

| Piece | Tool |
|---|---|
| Transcription | faster-whisper (`tiny` model, CPU) |
| Embeddings | FastEmbed (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| API | FastAPI |
| Frontend | React (see [lecture-search-frontend](https://github.com/longanillaralph/lecture-search-frontend)) |
