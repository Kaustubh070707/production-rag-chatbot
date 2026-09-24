from app.chunking import chunk_text

def test_chunking_overlap():
    t = "a " * 500
    chunks = chunk_text(t, size=800, overlap=80)
    assert len(chunks) >= 1
    assert all(len(c) <= 800 for c in chunks)

def test_chunking_short():
    assert chunk_text("hello", size=800, overlap=80) == ["hello"]
    assert chunk_text("", size=800, overlap=80) == []

def test_health():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/").status_code == 200

def test_refusal_gate():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    # unanswerable should refuse with empty citations, no LLM spend
    r = c.post("/ask", json={"query": "What is refund policy?", "top_k": 2})
    assert r.status_code == 200
    j = r.json()
    assert j["answer"] == "Not found in your documents."
    assert j["citations"] == []
