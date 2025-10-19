Graph-RAG Sprint
=================

Quickstart
----------

1) Create a virtual environment and install requirements:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

2) Copy environment defaults:

    copy .env.example .env

3) Run API:

    uvicorn backend.api:app --reload --port 8001

4) Run UI:

    streamlit run app/streamlit_app.py --server.port 8501

Docker
------

This launches two services: API (8001) and UI (8501).

    docker compose up --build

Testing
-------

    python -m pytest -q

Release
-------

Tag a release:

    git tag v0.1.0 -m "v0.1.0"
    git push origin v0.1.0

Troubleshooting Docker
----------------------

If Ollama is used, ensure OLLAMA_HOST is reachable from containers (e.g., set OLLAMA_HOST to a host-accessible address).


