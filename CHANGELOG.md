# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.2.0] - 2025-12-09
### Added - Major Feature Update
- **Neo4j Integration**: Complete Neo4j graph database integration
  - `backend/services/neo4j_store.py` with sub-graph retrieval
  - k-hop neighborhood queries
  - Shortest path finding between entities
  - `scripts/load_neo4j.py` for TSV to Neo4j data loading
- **GNN-Based Retrieval**: PyTorch Geometric implementation
  - `backend/services/gnn_retriever.py` with 2-layer GCN
  - Link prediction pre-training
  - Path scoring based on graph structure, centrality, and relevance
  - `scripts/train_gnn.py` for training GNN embeddings
  - Hybrid retrieval: BM25 + GNN structural scoring
- **Enhanced Source Tracing**: Proper citation extraction
  - Citations: `[Path 1]`, `[Path 2]` in LLM responses
  - Citation parsing and validation
  - Full reasoning trace with 4-step pipeline visualization
  - Path-to-answer mapping
- **Enhanced Streamlit Dashboard**:
  - Interactive graph visualization with Plotly
  - Cited paths highlighted in red
  - GNN confidence scores displayed
  - Step-by-step reasoning pipeline visualization
  - System health indicators (API, Neo4j, GNN)
- **Docker Compose Stack**: 3-service architecture
  - Neo4j (ports 7474, 7687)
  - Ollama (port 11434)
  - FastAPI + Streamlit (ports 8010, 8501)
  - Health checks and dependency ordering
- **Setup & Documentation**:
  - `scripts/setup.ps1` - Windows setup automation
  - `.env.example` - Environment configuration template
  - Comprehensive README.md with architecture diagrams
  - `IMPLEMENTATION_SUMMARY.md` - Feature verification

### Changed
- **Updated `backend/services/paths.py`**: Integrated GNN scoring into path retrieval
- **Updated `backend/services/rag.py`**: Added citation extraction and reasoning trace
- **Updated `requirements.txt`**: Added neo4j, torch, torch-geometric, plotly
- **Updated `docker-compose.yml`**: Expanded to 3-service stack with Neo4j and Ollama
- **Updated `app/streamlit_app.py`**: Complete redesign with visualization and citations

### Removed - Cleanup
- Deleted all `.bak` backup files (7 files)
- Removed `app/backend/` - duplicate implementation
- Removed redundant backend files:
  - `backend/brain.py`
  - `backend/composer.py`
  - `backend/server.py`
  - `backend/kg_loader.py`
  - `backend/retriever.backup.py`
- Removed research scripts:
  - `src/retriever_baseline.py`
  - `src/retriever_edgeaware.py`
  - `src/gen_toy.py`
  - `src/eval_pathk.py`
  - `src/graph_rag_pipeline.py`
- Removed test data files:
  - `baseline_paths.jsonl`
  - `edgeaware_paths.jsonl`
  - `project_structure.txt`
  - `lite-llm.yaml`

### Dependencies Added
- neo4j==5.14.0
- torch==2.1.0
- torch-geometric==2.4.0
- sentence-transformers==2.2.2
- scikit-learn==1.3.2
- plotly==5.18.0

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


