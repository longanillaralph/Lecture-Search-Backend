"""Reusable lecture ingestion and retrieval pipeline.

The command-line scripts and the HTTP API both use this module so that a
lecture is always identified by its YouTube video ID instead of by the
experimental lecture's filenames or collection names.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import parse_qs, urlparse

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Embedding
from chromadb.errors import NotFoundError
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
LECTURE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,63}$")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
    "www.youtu.be",
}

_embedding_model: Any | None = None
_embedding_model_lock = threading.Lock()
_whisper_model: Any | None = None
_whisper_model_lock = threading.Lock()


class LectureInputError(ValueError):
    """Raised when a submitted URL or lecture payload is invalid."""


class LLMNotConfigured(RuntimeError):
    """Raised when answer generation has not been configured for deployment."""


class LLMRequestError(RuntimeError):
    """Raised when the configured LLM cannot generate an answer."""


@dataclass(frozen=True)
class YouTubeSource:
    video_id: str
    canonical_url: str


@dataclass(frozen=True)
class IndexResult:
    lecture_id: str
    collection_name: str
    chunk_count: int
    already_indexed: bool


def normalize_youtube_url(raw_url: str) -> YouTubeSource:
    """Validate a single-video YouTube URL and return its canonical identity."""

    if not isinstance(raw_url, str) or not raw_url.strip():
        raise LectureInputError("A YouTube URL is required.")

    value = raw_url.strip()
    try:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except ValueError as exc:
        raise LectureInputError("The submitted URL is malformed.") from exc

    if parsed.scheme not in {"http", "https"} or hostname not in YOUTUBE_HOSTS:
        raise LectureInputError("The URL must be a supported YouTube video URL.")

    video_id: str | None = None
    if hostname in {"youtu.be", "www.youtu.be"}:
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            video_id = path_parts[0]
    else:
        query_video_id = parse_qs(parsed.query).get("v", [None])[0]
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts and path_parts[0] == "watch":
            video_id = query_video_id
        elif len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live", "v"}:
            video_id = path_parts[1]
        else:
            video_id = query_video_id

    if not video_id or not VIDEO_ID_RE.fullmatch(video_id):
        raise LectureInputError("The URL does not contain a valid single YouTube video ID.")

    return YouTubeSource(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
    )


def validate_lecture_id(lecture_id: str) -> str:
    value = str(lecture_id).strip()
    if not LECTURE_ID_RE.fullmatch(value):
        raise LectureInputError("The lecture_id contains unsupported characters.")
    return value


def collection_name_for(lecture_id: str, window_seconds: int = 30) -> str:
    lecture_id = validate_lecture_id(lecture_id)
    if window_seconds <= 0:
        raise LectureInputError("window_seconds must be positive.")
    name = f"lecture_{lecture_id}_{window_seconds}s"
    if len(name) > 63:
        raise LectureInputError("The lecture_id is too long for a Chroma collection name.")
    return name


def clean_transcript(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Whisper segments while preserving their timestamps."""

    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        cleaned.append({"start": start, "end": end, "text": text})
    return cleaned


def chunk_by_time(
    segments: Sequence[dict[str, Any]], window_seconds: int = 30
) -> list[dict[str, Any]]:
    """Group cleaned transcript segments into timestamped context windows."""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not segments:
        return []

    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if current is None:
            current = {"start": start, "end": end, "text": segment["text"]}
            continue

        if start - current["start"] > window_seconds:
            chunks.append(current)
            current = {"start": start, "end": end, "text": segment["text"]}
        else:
            current["end"] = max(float(current["end"]), end)
            current["text"] = f"{current['text']} {segment['text']}"

    if current is not None:
        chunks.append(current)
    return chunks


def load_transcript(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as transcript_file:
        raw_segments = json.load(transcript_file)
    if not isinstance(raw_segments, list):
        raise ValueError("A transcript must contain a JSON list of segments.")
    return clean_transcript(raw_segments)


def save_transcript(segments: Sequence[dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as transcript_file:
        json.dump(list(segments), transcript_file, indent=2, ensure_ascii=False)


def get_embedding_model() -> Any:
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                from fastembed import TextEmbedding

                model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                _embedding_model = TextEmbedding(model_name)
    return _embedding_model


def embed_texts(texts: Sequence[str], batch_size: int = 128) -> list[Embedding]:
    model = get_embedding_model()
    embeddings: list[Embedding] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(vector for vector in model.embed(batch))
    return embeddings


def get_chroma_client() -> ClientAPI:
    db_path = Path(os.getenv("CHROMA_DB_PATH", str(Path(__file__).resolve().parent / "chroma_db")))
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def get_lecture_collection(client: ClientAPI, lecture_id: str, window_seconds: int = 30) -> Any:
    name = collection_name_for(lecture_id, window_seconds)
    try:
        return client.get_collection(name)
    except NotFoundError as exc:
        raise LectureInputError(f"Lecture '{lecture_id}' has not been processed yet.") from exc


def index_lecture_chunks(
    client: ClientAPI,
    lecture_id: str,
    source: YouTubeSource,
    title: str,
    chunks: Sequence[dict[str, Any]],
    window_seconds: int = 30,
) -> IndexResult:
    """Create a per-lecture Chroma collection and add all chunk embeddings."""

    lecture_id = validate_lecture_id(lecture_id)
    if not chunks:
        raise ValueError("The transcript did not contain any searchable text.")

    collection_name = collection_name_for(lecture_id, window_seconds)
    metadata = {
        "lecture_id": lecture_id,
        "video_id": source.video_id,
        "source_url": source.canonical_url,
        "title": title or "",
        "window_seconds": window_seconds,
        "status": "processing",
    }

    try:
        collection = client.get_collection(collection_name)
        existing_metadata = collection.metadata or {}
        existing_count = collection.count()
        if existing_count > 0 and existing_metadata.get("status") == "ready":
            return IndexResult(lecture_id, collection_name, existing_count, True)
        if existing_count > 0:
            client.delete_collection(collection_name)
            collection = client.create_collection(collection_name, metadata=metadata)
    except NotFoundError:
        collection = client.create_collection(collection_name, metadata=metadata)

    try:
        texts = [str(chunk["text"]) for chunk in chunks]
        embeddings = embed_texts(texts)
        for start in range(0, len(chunks), 128):
            batch_chunks = chunks[start : start + 128]
            collection.add(
                ids=[f"{lecture_id}-{index}" for index in range(start, start + len(batch_chunks))],
                embeddings=embeddings[start : start + len(batch_chunks)],
                documents=[chunk["text"] for chunk in batch_chunks],
                metadatas=[
                    {
                        "lecture_id": lecture_id,
                        "video_id": source.video_id,
                        "source_url": source.canonical_url,
                        "title": title or "",
                        "start": float(chunk["start"]),
                        "end": float(chunk["end"]),
                    }
                    for chunk in batch_chunks
                ],
            )
        collection.modify(metadata={**metadata, "status": "ready", "chunk_count": len(chunks)})
    except Exception:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        raise

    return IndexResult(lecture_id, collection_name, len(chunks), False)


def download_youtube_audio(source: YouTubeSource, output_dir: str | Path) -> tuple[Path, str]:
    """Download the source audio without creating a permanent media file."""

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required to process YouTube URLs.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    options: Any = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path / "lecture.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }

    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(source.canonical_url, download=True)
        title = str(info.get("title") or source.video_id)

    audio_files = [
        path
        for path in output_path.glob("lecture.*")
        if path.is_file() and path.suffix not in {".part", ".ytdl", ".json"}
    ]
    if not audio_files:
        raise RuntimeError("yt-dlp completed without producing an audio file.")
    return audio_files[0], title


def get_whisper_model() -> Any:
    global _whisper_model
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel

                device = os.getenv("WHISPER_DEVICE", "cpu")
                default_compute_type = "int8" if device == "cpu" else "float16"
                compute_type = os.getenv("WHISPER_COMPUTE_TYPE", default_compute_type)
                model_name = os.getenv("WHISPER_MODEL", "tiny")
                _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _whisper_model


def transcribe_audio(audio_path: str | Path) -> list[dict[str, Any]]:
    model = get_whisper_model()
    segments, _ = model.transcribe(str(audio_path), word_timestamps=False)
    raw_segments = [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
    ]
    return clean_transcript(raw_segments)


def search_lecture(
    client: ClientAPI,
    lecture_id: str,
    question: str,
    top_k: int = 3,
    window_seconds: int = 30,
) -> list[dict[str, Any]]:
    question = " ".join(str(question).split())
    if not question:
        raise LectureInputError("A question is required.")
    if top_k < 1 or top_k > 20:
        raise LectureInputError("top_k must be between 1 and 20.")

    collection = get_lecture_collection(client, lecture_id, window_seconds)
    query_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    sources: list[dict[str, Any]] = []
    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        sources.append(
            {
                "lecture_id": metadata.get("lecture_id", lecture_id),
                "source_url": metadata.get("source_url"),
                "start": float(metadata.get("start", 0)),
                "end": float(metadata.get("end", 0)),
                "text": document,
            }
        )
    return sources


def generate_answer(question: str, sources: Sequence[dict[str, Any]]) -> str:
    """Generate a grounded answer through a configurable chat-completions API."""

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMNotConfigured(
            "Set LLM_API_KEY (or OPENAI_API_KEY) to enable answer generation."
        )

    endpoint = os.getenv(
        "LLM_CHAT_COMPLETIONS_URL", "https://api.openai.com/v1/chat/completions"
    )
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    context = "\n\n".join(
        f"[{source['start']:.1f}s–{source['end']:.1f}s]\n{source['text']}"
        for source in sources
    )
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the supplied lecture excerpts. "
                        "If the excerpts do not contain enough information, say so clearly. "
                        "Do not invent facts or citations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nLecture excerpts:\n{context}",
                },
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMRequestError("The configured LLM could not generate an answer.") from exc

    try:
        answer = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMRequestError("The configured LLM returned an invalid response.") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise LLMRequestError("The configured LLM returned an empty answer.")
    return answer.strip()


def ingest_youtube_lecture(
    client: ClientAPI,
    raw_url: str,
    window_seconds: int = 30,
) -> tuple[IndexResult, YouTubeSource, str]:
    """Download, transcribe, clean, chunk, embed, and index one lecture."""

    source = normalize_youtube_url(raw_url)
    lecture_id = source.video_id
    collection_name = collection_name_for(lecture_id, window_seconds)
    try:
        existing = client.get_collection(collection_name)
        existing_metadata = existing.metadata or {}
        existing_count = existing.count()
        if existing_count > 0 and existing_metadata.get("status") == "ready":
            metadata = existing_metadata
            return (
                IndexResult(lecture_id, collection_name, existing_count, True),
                source,
                str(metadata.get("title") or lecture_id),
            )
    except NotFoundError:
        pass

    with tempfile.TemporaryDirectory(prefix=f"lecture-{lecture_id}-") as temp_dir:
        audio_path, title = download_youtube_audio(source, temp_dir)
        transcript = transcribe_audio(audio_path)
        chunks = chunk_by_time(transcript, window_seconds=window_seconds)
        result = index_lecture_chunks(
            client,
            lecture_id=lecture_id,
            source=source,
            title=title,
            chunks=chunks,
            window_seconds=window_seconds,
        )
    return result, source, title
