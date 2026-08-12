import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.graph import run_query

app = FastAPI(title="GroundedRAG")

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(req: QueryRequest):
    # No app-level cache here on purpose -- caching happens one layer down,
    # at the Portkey gateway, on every LLM call this makes. Ask the same
    # question twice in the demo and the second one comes back near-instant.
    result = run_query(req.query)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
