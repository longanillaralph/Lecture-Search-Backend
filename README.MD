# Lecture Search Tool

Upload a lecture recording, get back a searchable transcript. Ask a question about the lecture in plain language and jump straight to the timestamp where it was discussed — instead of scrubbing through a 75-minute recording by hand.

## How it works

The pipeline is six scripts, run in order. Each one reads the file(s) the previous script produced.

1. **Download audio** — grab a lecture recording from YouTube using `yt-dlp` (or use any local audio/video file).
2. **`transcribe.py`** — loads the recording and runs it through Whisper (`faster-whisper`) to produce a timestamped transcript. Output: `transcript.json`, a list of small segments (mostly single sentences), each with a start time, end time, and text.
3. **`chunk.py`** — groups those small segments into bigger windows, so each piece has enough context to make sense on its own. Generates three window sizes to compare: `chunks_30s.json`, `chunks_60s.json`, `chunks_120s.json` (see "Chunk size evaluation" below for why).
4. **`index.py`** — turns each chunk's text into an embedding (a list of numbers representing its meaning) using a sentence-transformers model, then stores the embedding, the original text, and the timestamp together in a Chroma vector database. Each chunk size gets its own collection (`lecture1_30s`, `lecture1_60s`, `lecture1_120s`). Output: a `chroma_db/` folder on disk (Chroma uses SQLite internally for persistence, but all reads/writes go through Chroma's own API, not raw SQL).
5. **`search.py`** — loads the same embedding model, takes a typed question, turns it into an embedding the same way, and asks Chroma for the 3 stored chunks whose embeddings are closest in meaning. Prints each result's timestamp range and text.
6. **`evaluate.py`** — runs a set of test questions (each with a known correct timestamp) against every chunk size and measures how often the correct result comes back. This is the script that produced the evidence behind the chunk-size decision below.

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
              v
     top 3 relevant timestamps
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
- 2 of 7 questions failed at every window size. The corresponding timestamps were manually re-verified as correct, ruling out chunking as the cause — this points instead to a limitation in the `tiny` Whisper model's transcription accuracy or the embedding model itself, worth investigating separately.
- **Selected: 30-second windows.** With 30s and 60s tied on accuracy, the tie-breaker was timestamp precision — a 30-second window lands a user closer to the exact moment they're looking for than a 60-second one, which matters more once accuracy is no longer the deciding factor.

## Setup

```bash
uv venv --python 3.12
uv pip install faster-whisper sentence-transformers chromadb streamlit yt-dlp
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

# 6. Search (uses the 30s window, per the evaluation above)
python search.py
```

## Current status

- [x] Transcription pipeline working (tested on a 75-minute calculus lecture, 1618 raw segments)
- [x] Chunking into three window sizes (30s: 157 chunks, 60s: 82 chunks, 120s: 43 chunks)
- [x] Embedding + Chroma indexing (three separate collections)
- [x] Semantic search returning relevant, correctly-distinguished results across different topics in the same lecture
- [x] Compared chunk window sizes against 7 labeled test questions — **30-second windows selected**, based on accuracy matching the 60s window while returning more precise timestamps
- [ ] Streamlit interface (upload + search box)
- [ ] Deploy

## Resume bullet

Built a lecture search tool that transcribes audio and enables semantic search over timestamped segments; evaluated multiple chunking strategies against retrieval quality and selected 30-second windows, chosen for retrieval accuracy on par with larger windows while returning more precise timestamps.

## Stack

| Piece | Tool |
|---|---|
| Transcription | faster-whisper (`tiny` model, CPU) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| Interface (planned) | Streamlit |