FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install uvicorn[standard] streamlit

COPY . /app

ENV GRAPH_EDGES_PATH=/app/data/edges.tsv \
    GRAPH_NODE_TEXTS_PATH=/app/data/node_texts.jsonl \
    API_URL=http://localhost:8001

EXPOSE 8001 8501

CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port 8001 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]


