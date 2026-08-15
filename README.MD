# Lecture Search Tool — Backend

Upload a lecture recording, get back a searchable transcript. Ask a question about the lecture in plain language and jump straight to the timestamp where it was discussed — instead of scrubbing through a 75-minute recording by hand.

This is the backend: transcription, chunking, embedding, search, and evaluation. It exposes both a console interface and a FastAPI HTTP API. For the React frontend that talks to this API, see [lecture-search-frontend](https://github.com/longanillaralph/lecture-search-frontend).

## How it works

The pipeline is six scripts, run in order. Each one reads the file(s) the previous script produced. Two additional scripts (`search.py` and `api.py`) offer alternative ways to query the finished index — a console prompt and an HTTP API respectively.

1. **Download audio** — grab a lecture recording from YouTube using `yt-dlp` (or use any local audio/video file).
2. **`transcribe.py`** — loads the recording and runs it through Whisper (`faster-whisper`) to produce a timestamped transcript. Output: `transcript.json`.
3. **`chunk.py`** — groups the transcript into bigger windows. Generates three sizes to compare: `chunks_30s.json`, `chunks_60s.json`, `chunks_120s.json`.
4. **`index.py`** — embeds each chunk with `sentence-transformers` and stores it in a Chroma vector database. Each chunk size gets its own collection (`lecture1_30s`, `lecture1_60s`, `lecture1_120s`).
5. **`evaluate.py`** — runs labeled test questions against every chunk size and measures retrieval accuracy, producing the evidence behind the chunk-size decision (see below).
6. **Query the index** — either:
   - **`search.py`** — console prompt, prints top 3 results with timestamps.
   - **`api.py`** — FastAPI HTTP endpoint (`GET /search?q=...`), returns structured JSON. Includes CORS support for the React frontend running on `localhost:5173`. Interactive docs at `/docs`.

```
YouTube link
     |
     v
  yt-dlp  ---->  audio file (.webm/.mp3)
                       |
                       v
                transcribe.py  ---->  transcript.json
                       |
                       v
                  chunk.py      ---->  chunks_30s.json / chunks_60s.json / chunks_120s.json
                       |
                       v
                  index.py      ---->  chroma_db/ (3 collections)
                       |
                       v
         search.py  <----  evaluate.py (tests all 3 sizes)
              |
       (console)
              |
         api.py  ---->  (HTTP JSON, used by the React frontend)
```

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
uv pip install faster-whisper sentence-transformers chromadb streamlit yt-dlp fastapi uvicorn
```

## Usage

```bash
# 1. Download a lecture (or supply your own audio/video file)
yt-dlp -x --audio-format mp3 "YOUTUBE_URL"

# 2. Transcribe (edit the filename inside transcribe.py first)
python transcribe.py

# 3. Chunk the transcript into three window sizes
python chunk.py

# 4. Build the searchable index (all three chunk sizes)
python index.py

# 5. Evaluate which chunk size performs best
python evaluate.py

# 6a. Search from the console
python search.py

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
- [ ] Support multiple lectures (currently the `lecture1` collection prefix is hardcoded — see DOCUMENTATION.md Known Limitations)
- [ ] Deploy backend + frontend

## Resume bullet

Built a lecture search tool that transcribes audio and enables semantic search over timestamped segments; evaluated multiple chunking strategies against retrieval quality and selected 30-second windows, chosen for retrieval accuracy on par with larger windows while returning more precise timestamps.

## Stack

| Piece | Tool |
|---|---|
| Transcription | faster-whisper (`tiny` model, CPU) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| API | FastAPI |
| Frontend | React (see [lecture-search-frontend](https://github.com/longanillaralph/lecture-search-frontend)) |