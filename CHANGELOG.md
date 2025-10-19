# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.1.0] - 2025-10-19
### Added
- GET /models endpoint to list Ollama models.
- dry_run flag on POST /ask to skip LLM.
- CORS for http://localhost:8501 (dev).
- In-memory LRU cache for /ask responses.
- Structured JSON logging and LOG_LEVEL env support.
- Streamlit: model dropdown, dry run, path limit, copy request.
- Docker Compose with separate API and UI services and healthcheck.
- Tests for models endpoint, provider fallback, dry-run, smoke /ask.

### Changed
- Canonicalized FastAPI entrypoint to backend/main.py.
- Centralized retrieval in backend/retrieval/core.py.
- Answer generation via backend/models/provider.py with Ollama and fallback.

### Fixed
- Trimmed path_scores consistently after ranking caps.
- Import collisions by re-exporting app from backend/api/__init__.py.


