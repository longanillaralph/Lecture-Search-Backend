"""FastAPI entrypoint for dynamic, per-lecture search."""

from __future__ import annotations

import os
import threading
from typing import Any

from chromadb.errors import NotFoundError
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AnyHttpUrl, BaseModel, Field

from pipeline import (
    LLMNotConfigured,
    LLMRequestError,
    LectureInputError,
    collection_name_for,
    generate_answer,
    get_chroma_client,
    ingest_youtube_lecture,
    search_lecture,
)


app = FastAPI(title="Lecture Search API", version="2.0.0")


DEFAULT_CORS_ORIGINS = [
    "https://lecture-search-frontend.vercel.app",
    "http://localhost:5173",
]

cors_env = os.getenv("CORS_ORIGINS", "")
configured_origins = [
    origin.strip().strip('"').strip("'")
    for origin in cors_env.split(",")
    if origin.strip()
]

allowed_origins = configured_origins or DEFAULT_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


client = get_chroma_client()
ingestion_lock = threading.Lock()


class LectureCreateRequest(BaseModel):
    url: AnyHttpUrl = Field(
        ...,
        description="A single supported YouTube video URL",
    )

    model_config = {"populate_by_name": True}


class LectureResponse(BaseModel):
    lecture_id: str
    video_id: str
    title: str
    source_url: str
    status: str
    chunk_count: int


class LectureDetailsResponse(LectureResponse):
    collection_name: str


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, alias="q")
    lecture_id: str | None = None
    url: AnyHttpUrl | None = None
    top_k: int = Field(default=3, ge=1, le=20)
    generate_answer: bool = True

    model_config = {"populate_by_name": True}


class ProcessedLectureSearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, alias="q")
    top_k: int = Field(default=3, ge=1, le=20)
    generate_answer: bool = True

    model_config = {"populate_by_name": True}


class SearchResult(BaseModel):
    lecture_id: str
    source_url: str | None = None
    start: float
    end: float
    text: str


class SearchResponse(BaseModel):
    query: str
    lecture_id: str
    answer: str | None = None
    results: list[SearchResult]


@app.get("/healthz")
def health_check() -> dict[str, str]:
    """Small readiness endpoint for a hosting provider or load balancer."""

    return {"status": "ok"}


def _http_error_for_pipeline(exc: Exception) -> HTTPException:
    if isinstance(exc, LectureInputError):
        return HTTPException(status_code=400, detail=str(exc))

    if isinstance(exc, LLMNotConfigured):
        return HTTPException(status_code=503, detail=str(exc))

    if isinstance(exc, LLMRequestError):
        return HTTPException(status_code=502, detail=str(exc))

    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=502, detail=str(exc))

    return HTTPException(
        status_code=500,
        detail="Lecture processing failed.",
    )


def _process_lecture(url: str) -> LectureResponse:
    try:
        with ingestion_lock:
            result, source, title = ingest_youtube_lecture(client, url)

    except Exception as exc:
        raise _http_error_for_pipeline(exc) from exc

    return LectureResponse(
        lecture_id=result.lecture_id,
        video_id=source.video_id,
        title=title,
        source_url=source.canonical_url,
        status="already_indexed" if result.already_indexed else "ready",
        chunk_count=result.chunk_count,
    )


def _answer_if_requested(
    question: str,
    results: list[dict[str, Any]],
    requested: bool,
) -> str | None:
    if not requested:
        return None

    try:
        return generate_answer(question, results)

    except Exception as exc:
        raise _http_error_for_pipeline(exc) from exc


def _search_response(
    question: str,
    lecture_id: str,
    top_k: int,
    with_answer: bool,
) -> SearchResponse:
    try:
        results = search_lecture(
            client,
            lecture_id,
            question,
            top_k=top_k,
        )

    except Exception as exc:
        raise _http_error_for_pipeline(exc) from exc

    return SearchResponse(
        query=question,
        lecture_id=lecture_id,
        answer=_answer_if_requested(
            question,
            results,
            with_answer,
        ),
        results=[
            SearchResult(**result)
            for result in results
        ],
    )


@app.post(
    "/lectures",
    response_model=LectureResponse,
    status_code=201,
)
def create_lecture(
    request: LectureCreateRequest,
) -> LectureResponse:
    """Process a YouTube lecture and return its stable lecture_id."""

    return _process_lecture(str(request.url))


@app.get(
    "/lectures/{lecture_id}",
    response_model=LectureDetailsResponse,
)
def get_lecture(
    lecture_id: str,
) -> LectureDetailsResponse:
    """Return metadata for an indexed lecture."""

    try:
        collection = client.get_collection(
            collection_name_for(lecture_id)
        )

    except LectureInputError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except NotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Lecture has not been processed yet.",
        ) from exc

    metadata = collection.metadata or {}

    return LectureDetailsResponse(
        lecture_id=str(
            metadata.get("lecture_id", lecture_id)
        ),
        video_id=str(
            metadata.get("video_id", lecture_id)
        ),
        title=str(metadata.get("title", "")),
        source_url=str(
            metadata.get("source_url", "")
        ),
        status=str(
            metadata.get("status", "ready")
        ),
        chunk_count=int(
            metadata.get(
                "chunk_count",
                collection.count(),
            )
        ),
        collection_name=collection.name,
    )


@app.post(
    "/lectures/{lecture_id}/search",
    response_model=SearchResponse,
)
def search_processed_lecture(
    lecture_id: str,
    request: ProcessedLectureSearchRequest,
) -> SearchResponse:
    """Search only the requested lecture."""

    return _search_response(
        question=request.question,
        lecture_id=lecture_id,
        top_k=request.top_k,
        with_answer=request.generate_answer,
    )


@app.post(
    "/search",
    response_model=SearchResponse,
)
def search_request(
    request: SearchRequest,
) -> SearchResponse:
    """Search an indexed lecture or process a URL first."""

    lecture_id = request.lecture_id

    if request.url is not None:
        lecture = _process_lecture(str(request.url))
        lecture_id = lecture.lecture_id

    if not lecture_id:
        raise HTTPException(
            status_code=400,
            detail="lecture_id or url is required.",
        )

    return _search_response(
        question=request.question,
        lecture_id=lecture_id,
        top_k=request.top_k,
        with_answer=request.generate_answer,
    )


@app.get("/search", response_model=SearchResponse)
def legacy_search(
    q: str = Query(..., min_length=1),
    lecture_id: str | None = None,
    url: AnyHttpUrl | None = None,
    youtube_url: AnyHttpUrl | None = None,
    top_k: int = Query(default=3, ge=1, le=20),
) -> SearchResponse:
    """Backward-compatible GET search route."""

    submitted_url = url or youtube_url

    if submitted_url is not None:
        lecture = _process_lecture(str(submitted_url))
        lecture_id = lecture.lecture_id

    if lecture_id:
        return _search_response(
            question=q,
            lecture_id=lecture_id,
            top_k=top_k,
            with_answer=False,
        )

    raise HTTPException(
        status_code=400,
        detail=(
            "lecture_id or url is required; "
            "searches are scoped to one lecture."
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
