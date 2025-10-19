﻿import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    j = r.json()
    assert "name" in j and "version" in j
