# Lecture Search — Next Features and Database Plan

This roadmap describes what to add next and when a database becomes worthwhile.

## Recommended priority

### Phase 1 — Production reliability

- [ ] Persist processing jobs instead of keeping them only in memory.
- [ ] Add job states: `queued`, `processing`, `ready`, `failed`.
- [ ] Store the error message and last update time for failed jobs.
- [ ] Add a retry button for failed processing.
- [ ] Add backend rate limiting.
- [ ] Add structured logs with a request ID and lecture ID.
- [ ] Add tests for captions, indexing, polling, CORS, and LLM errors.

This phase should come before adding many user-facing features because a restart currently loses active job state.

### Phase 2 — Add a relational database

Add PostgreSQL for application data, not as a direct replacement for Chroma at first.

Suggested tables:

```text
users
  id, email, created_at

lectures
  id, video_id, source_url, title, status, error_message,
  chunk_count, created_at, updated_at, completed_at

processing_jobs
  id, lecture_id, status, attempts, started_at, finished_at, error_message

questions
  id, lecture_id, user_id, question, answer, created_at

question_sources
  id, question_id, start_seconds, end_seconds, text, source_url
```

PostgreSQL should own users, lectures, jobs, questions, and usage data. Chroma should continue to own vector embeddings until a later migration is justified.

### Phase 3 — Improve the lecture workspace

- [ ] Show the real YouTube title and thumbnail.
- [ ] Display lecture duration and processing time.
- [ ] Add a transcript browser with search and timestamp navigation.
- [ ] Add a “copy answer” button.
- [ ] Add collapsible source excerpts.
- [ ] Add question history persistence.
- [ ] Add an option to regenerate an answer.
- [ ] Add a clear lecture/delete action.

### Phase 4 — Accounts and limits

- [ ] Add authentication.
- [ ] Associate lectures with users.
- [ ] Add private/public lecture visibility.
- [ ] Add per-user lecture and question limits.
- [ ] Add provider usage tracking and cost controls.
- [ ] Add an admin view for failed jobs and usage.

## Database decision

### Option A: PostgreSQL + Chroma volume — recommended next step

Use PostgreSQL for metadata and job state while keeping Chroma on a persistent volume.

Advantages:

- Smallest code change.
- Reliable job recovery after backend restarts.
- Easy user, lecture, and question relationships.
- Keeps the existing vector search implementation.

Tradeoff: there are two persistence systems to operate.

### Option B: PostgreSQL with pgvector

Move embeddings from ChromaDB into PostgreSQL using the `pgvector` extension.

Advantages:

- One database for metadata and vectors.
- Easier backups and ownership relationships.
- Better foundation for multi-user production systems.

Tradeoffs:

- Requires rewriting indexing and search code.
- Requires vector indexes and query tuning.
- More migration and operational work.

### Option C: Managed vector database

Use a managed vector provider and PostgreSQL for application metadata.

Advantages:

- Scales independently from the API.
- Avoids local Chroma volume management.

Tradeoffs:

- Adds another paid service and API dependency.
- Requires provider-specific integration.

## Recommended implementation order for PostgreSQL

1. Add a PostgreSQL service.
2. Add a migration tool such as Alembic.
3. Create `lectures` and `processing_jobs` tables.
4. Write a job row before starting background processing.
5. Update the row after each state transition.
6. On startup, mark abandoned `processing` jobs as `failed` or requeue them.
7. Return job state from `GET /lectures/{lecture_id}`.
8. Add `questions` and `question_sources` after job persistence works.

## Suggested job state machine

```text
queued → processing → ready
                  ↘ failed → queued (retry)
```

Every transition should update `updated_at`. A job should have a maximum retry count so a broken video or provider does not loop forever.

## What not to add yet

- Do not add multiple LLM providers to the UI before provider errors are observable in logs.
- Do not migrate away from Chroma until job persistence and backups are working.
- Do not expose API keys to the frontend.
- Do not add real-time websockets before polling is reliable and tested.
- Do not add large embedding models while memory is limited.

## Definition of a production-ready next version

- Jobs survive service restarts.
- A failed lecture shows a useful error and can be retried.
- Chroma data is backed up or replaceable.
- Users can see only their own lectures.
- Provider keys are rotated and stored as secrets.
- Questions and answers can be audited.
- Automated tests cover the full processing and search flow.
