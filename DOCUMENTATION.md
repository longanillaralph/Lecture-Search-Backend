# Lecture Search Tool — Technical Documentation

> This document is the deep technical reference: exact code walkthroughs, data schemas, and known limitations. For a quick overview of what the project does and the chunk-size evaluation results, see [`README.MD`](./README.MD).

## Project Overview

The **Lecture Search Tool** is a Python pipeline that converts lecture recordings into a searchable, timestamped knowledge base. A user supplies a lecture recording, and the system:

1. Transcribes the audio into text with timestamps
2. Groups the transcript into meaningful context chunks — at **three window sizes** (30s, 60s, 120s), to allow evidence-based comparison rather than guessing at one
3. Converts each chunk into a semantic embedding (a numerical vector representing meaning)
4. Stores the embeddings in a local vector database — one collection per chunk size
5. Evaluates all three chunk sizes against a set of labeled test questions to determine which performs best
6. Allows natural-language queries against the selected chunk size, returning the **top 3 most relevant timestamps** in the lecture

This eliminates the need to manually scrub through a 75-minute recording to find where a specific topic was covered.

---

## System Architecture

The pipeline consists of **six stages**. Stages 1–4 run once per lecture; stage 5 (evaluation) is run once to select a chunk size; stage 6 (search) is the ongoing end-user interaction.

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
                  index.py      ---->  chroma_db/ (3 collections: lecture1_30s, lecture1_60s, lecture1_120s)
                       |
                       v
         search.py  <----  evaluate.py (tests all 3 sizes against labeled questions)
              |
              v
     top 3 relevant timestamps
```

| Stage | Script | Input | Output |
|---|---|---|---|
| Download | `yt-dlp` (CLI) | YouTube URL | Audio file (`.webm` / `.mp3`) |
| Transcription | `transcribe.py` | Audio file | `transcript.json` |
| Chunking | `chunk.py` | `transcript.json` | `chunks_30s.json`, `chunks_60s.json`, `chunks_120s.json` |
| Indexing | `index.py` | `chunks_*.json` (all three) | `chroma_db/` — three collections |
| Evaluation | `evaluate.py` | `test_questions.json` + `chroma_db/` | Accuracy score per chunk size (console output) |
| Search | `search.py` | User query + `chroma_db/` (`lecture1_30s` collection) | Top 3 timestamped results |

---

## Pipeline Components

### 1. `transcribe.py` — Speech-to-Text

**Purpose:** Converts an audio/video lecture recording into a timestamped text transcript.

**Input:** An audio file passed directly to the Whisper model (the filename is currently hardcoded in the script).

**Output:** `transcript.json` — a list of segments, where each segment represents a short utterance (typically one sentence) with start/end timestamps.

**How it works:**

```python
model = WhisperModel("tiny", device="cpu", compute_type="int8")
```

- Uses the **`faster-whisper`** library with the **`tiny`** model.
- Runs on **CPU** with **int8 quantization** for speed and lower memory usage. (A GPU/CUDA path was attempted but blocked by a missing `cublas64_12.dll` dependency that a `pip install nvidia-cudnn-cu12` did not resolve — CPU was chosen deliberately to keep the pipeline unblocked. Worth revisiting later, since GPU would meaningfully speed up transcription of future lectures.)

```python
segments, info = model.transcribe("lecture.webm", word_timestamps=True)
```

- Transcribes the audio file, enabling word-level timestamps.

```python
results.append({"start": segment.start, "end": segment.end, "text": segment.text})
```

- Each segment is stored with:
  - `start` — start time in seconds (float)
  - `end` — end time in seconds (float)
  - `text` — the transcribed text

**Dependencies:** `faster-whisper`, `json`

---

### 2. `chunk.py` — Segment Chunking

**Purpose:** Groups the small transcription segments into larger, context-rich chunks — at **three window sizes** for comparison, rather than committing to a single size upfront.

**Input:** `transcript.json`

**Output:** `chunks_30s.json`, `chunks_60s.json`, `chunks_120s.json`

**Why chunking matters, and why three sizes:**
Individual sentence segments are too short to carry meaningful context. Grouping them into larger windows ensures each chunk has enough surrounding context to stand alone semantically. But the right window size isn't obvious upfront — too small loses context, too large returns more than the user wanted. Rather than picking one size by intuition, this pipeline generates all three and lets `evaluate.py` measure which actually performs best (see the README's "Chunk size evaluation" section for the results: 30s and 60s tied at 4/7 test questions correct, 120s underperformed at 2/7; 30s was selected as the tie-breaker for timestamp precision).

**How it works:**

```python
def chunk_by_time(segments, window_seconds=60):
```

- Iterates through segments, accumulating text until the elapsed time from the chunk's start exceeds `window_seconds`.
- When the window boundary is crossed, the current chunk is finalized and a new chunk begins.

```python
for window in [30, 60, 120]:
    chunks = chunk_by_time(segments, window_seconds=window)
    filename = f"chunks_{window}s.json"
```

- The same chunking function is run three times, once per window size, each writing to its own file.

**Chunk schema (identical across all three files, differing only in average chunk length):**

```json
{
  "start": 0.0,       // start time of the chunk (seconds)
  "end": 60.4,        // end time of the chunk (seconds)
  "text": "..."       // concatenated transcript text within the window
}
```

**Practical example:** A 75-minute lecture (1,618 raw segments) chunked down to:
- 157 chunks at 30-second windows
- 82 chunks at 60-second windows
- 43 chunks at 120-second windows

**Dependencies:** `json`

---

### 3. `index.py` — Embedding & Vector Indexing

**Purpose:** Converts each chunk's text into a **semantic embedding** and stores everything in a persistent **ChromaDB** vector database — separately for each chunk size.

**Input:** `chunks_30s.json`, `chunks_60s.json`, `chunks_120s.json`

**Output:** `chroma_db/` — a persistent local vector database folder containing three collections (Chroma uses SQLite under the hood for persistence, but all reads/writes go through Chroma's own API).

**How it works:**

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

for window in [30, 60, 120]:
    filename = f"chunks_{window}s.json"
    collection_name = f"lecture1_{window}s"
    collection = client.get_or_create_collection(collection_name)
```

- Uses the **`sentence-transformers`** library with the **`all-MiniLM-L6-v2`** model to produce embeddings.
- Loops over all three chunk sizes, creating a separate named collection for each (`lecture1_30s`, `lecture1_60s`, `lecture1_120s`) so they can be queried and evaluated independently.

```python
embeddings = model.encode(texts).tolist()
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    embeddings=embeddings,
    documents=texts,
    metadatas=[{"start": c["start"], "end": c["end"]} for c in chunks]
)
```

- Encodes all chunk texts into embeddings in one batch, per collection.
- Adds each chunk to its collection with `ids`, `embeddings`, `documents` (original text), and `metadatas` (timestamp range).

**Dependencies:** `chromadb`, `sentence-transformers`, `json`

---

### 4. `evaluate.py` — Chunk Size Evaluation

**Purpose:** Measures retrieval accuracy across all three chunk sizes against a labeled set of real questions, producing the evidence behind the chunk-size decision.

**Input:** `test_questions.json` (manually authored — 7 real questions about the lecture, each with a manually-verified correct timestamp) + `chroma_db/` (all three collections)

**Output:** Console printout of per-question hits/misses per chunk size, plus a final accuracy summary.

**How it works:**

```python
def is_correct(results, correct_start, tolerance=30):
    for meta in results["metadatas"][0]:
        if abs(meta["start"] - correct_start) <= tolerance:
            return True
    return False
```

- A returned result counts as correct if its start timestamp is within **30 seconds** of the manually-verified correct timestamp — chunks span time ranges rather than exact points, so a near-match is still a genuinely useful result for a student.

```python
for window in [30, 60, 120]:
    collection = client.get_collection(f"lecture1_{window}s")
    for item in test_questions:
        query_embedding = model.encode([item["question"]]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        if is_correct(results, item["correct_start"]):
            correct_count += 1
```

- For each chunk size, every test question is embedded and queried the same way `search.py` does, and scored against the known-correct timestamp.

**Result:** 30s and 60s windows both scored 4/7; 120s scored 2/7. Two questions failed at every window size — their timestamps were manually re-verified as correct, ruling out chunking as the cause and pointing instead to a limitation in the `tiny` Whisper model's transcription accuracy or the embedding model itself. 30-second windows were selected, using timestamp precision as the tie-breaker over the equally-accurate 60-second option.

**Dependencies:** `chromadb`, `sentence-transformers`, `json`

---

### 5. `search.py` — Semantic Search

**Purpose:** Accepts a natural-language question from the user and returns the **top 3 chunks** whose meaning is closest to the query, along with their timestamp ranges. Queries the `lecture1_30s` collection — the chunk size selected by `evaluate.py`.

**Input:** User query (via console prompt) + `chroma_db/`

**Output:** Top 3 results printed to the console, each showing its `[start → end]` timestamp range and the matching text.

**How it works:**

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("lecture1_30s")
```

- Re-loads the **same embedding model** used during indexing (critical: query and stored chunks must live in the same vector space).
- Opens the existing Chroma database and the selected `lecture1_30s` collection.

```python
query = input("Ask a question about the lecture: ")
query_embedding = model.encode([query]).tolist()
```

- Prompts the user for a natural-language question, encodes it into an embedding using the same model.

```python
results = collection.query(query_embeddings=query_embedding, n_results=3)
```

- Asks Chroma for the **3 nearest neighbors** in vector space (semantic similarity).

```python
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"\n[{meta['start']:.0f}s -> {meta['end']:.0f}s]")
    print(doc)
```

- Prints each result with its formatted timestamp range and the original chunk text.

**Dependencies:** `chromadb`, `sentence-transformers`

---

## Data Formats

### `transcript.json`

Produced by `transcribe.py`. An array of segment objects:

```json
[
  {
    "start": 0.0,
    "end": 3.5,
    "text": "Welcome to Calculus one lecture one point one."
  },
  {
    "start": 3.7,
    "end": 8.2,
    "text": "Today we're going to introduce the concept of limits."
  }
]
```

### `chunks_30s.json` / `chunks_60s.json` / `chunks_120s.json`

Produced by `chunk.py`. Same schema across all three files — an array of chunk objects, each with a longer text span. Files differ only in average chunk length and count (157 / 82 / 43 chunks respectively for the reference lecture):

```json
[
  {
    "start": 0.0,
    "end": 30.2,
    "text": "Welcome to Calculus one lecture one point one. Today we're going to introduce the concept of limits. ..."
  },
  {
    "start": 30.2,
    "end": 61.5,
    "text": "..."
  }
]
```

### `test_questions.json`

Authored manually. An array of question/timestamp pairs used by `evaluate.py`:

```json
[
  {"question": "when does she explain what a limit is", "correct_start": 796},
  {"question": "what is a tangent line", "correct_start": 135}
]
```

### `chroma_db/`

Produced by `index.py`. A persistent vector database directory managed by Chroma, containing three collections (`lecture1_30s`, `lecture1_60s`, `lecture1_120s`) — each with embeddings, original documents, metadata, and IDs.

---

## Setup & Installation

### Prerequisites

- Python 3.12 (pinned — `ctranslate2`, a `faster-whisper` dependency, does not yet ship wheels for 3.13)
- `uv` (package manager) or `pip`

### Installation

```bash
uv venv --python 3.12
uv pip install faster-whisper sentence-transformers chromadb streamlit yt-dlp
```

---

## Usage Guide

### 1. Download the lecture audio

```bash
yt-dlp -x --audio-format mp3 "YOUTUBE_URL"
```

> Alternatively, use any local audio/video file. Note: if `ffmpeg` is not installed, the mp3 conversion step will fail — `faster-whisper` (via PyAV) can read the raw `.webm` directly, so this is not a blocker.

### 2. Transcribe

Edit the filename inside `transcribe.py` to point to your audio file, then run:

```bash
python transcribe.py
```

### 3. Chunk the transcript into three window sizes

```bash
python chunk.py
```

### 4. Build the searchable index (all three chunk sizes)

```bash
python index.py
```

### 5. Evaluate which chunk size performs best

```bash
python evaluate.py
```

### 6. Search (uses the selected 30s window)

```bash
python search.py
```

You'll be prompted: `Ask a question about the lecture:` — type a natural-language question. The script prints the top 3 relevant chunks with their timestamp ranges:

```
[154s -> 215s]
So when we talk about the limit of a function, we're asking what value the function is approaching...
```

---

## How Semantic Search Works

Traditional keyword search matches **exact words**. This system performs **semantic search**, which matches based on **meaning**.

1. Each chunk of text is converted into an **embedding** — a high-dimensional vector of numbers that captures the semantic meaning of the text.
2. Texts with similar meaning are located **close together in vector space**, even if they use different words.
3. When a user asks a question, it is converted into an embedding using the **same model**, then vector similarity (distance) is computed against all stored chunk embeddings.
4. The chunks with the **smallest distance** (most similar meaning) are returned.

**Concrete example:** Asking *"what is limits?"* correctly returns chunks about left-side/right-side limits and limit existence — even though the exact word "limits" may not appear in every matched chunk.

---

## Project Status & Roadmap

### Completed

- [x] Transcription pipeline working (tested on a 75-minute Calculus lecture → 1,618 raw segments)
- [x] Chunking into three window sizes (30s: 157 chunks, 60s: 82 chunks, 120s: 43 chunks)
- [x] Embedding + ChromaDB indexing (three separate collections)
- [x] Semantic search working — returns correct and differentiated results for different topics within the same lecture
- [x] Chunk size evaluation against 7 labeled test questions — 30-second windows selected (see `evaluate.py` walkthrough above and README for full results table)

### Planned

- [ ] Streamlit interface (upload + search box)
- [ ] FastAPI backend (in progress — `api.py`)
- [ ] Deployment
- [ ] Investigate the two test questions that failed across all chunk sizes — likely a `tiny` Whisper transcription accuracy or embedding model limitation, not a chunking issue
- [ ] Support multiple lectures in one deployment (see Known Limitations below)

---

## Technology Stack

| Component | Tool |
|---|---|
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`tiny` model, CPU, int8) |
| Embeddings | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) (`all-MiniLM-L6-v2`) |
| Vector store | [ChromaDB](https://github.com/chroma-core/chroma) |
| Audio download | `yt-dlp` |
| Interface (planned) | Streamlit |
| API (planned) | FastAPI |

---

## Notes & Known Limitations

- The audio filename in `transcribe.py` is **hardcoded** — it must be edited manually for each new lecture.
- Collections are named per-lecture-and-window-size (`lecture1_30s`, etc.) but the `lecture1` prefix is itself hardcoded — indexing a second lecture would need a naming scheme change (e.g. deriving the prefix from the filename) or it will collide with the first lecture's collections.
- The `tiny` Whisper model prioritizes speed over accuracy; switching to `small`/`base` would improve transcription quality at the cost of runtime. A GPU path exists in principle but is currently blocked by an unresolved `cublas64_12.dll` dependency issue on this machine.
- The search is **console-based** in `search.py` — a Streamlit UI and FastAPI backend are both planned/in-progress (see Project Status above).
- The 30-second-window selection is based on a small test set (7 questions, one lecture). A larger and more diverse test set — more questions, multiple lectures — would give more confidence in the choice generalizing beyond this specific recording.