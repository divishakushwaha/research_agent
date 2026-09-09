from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from pipe import run_smart_search_and_store, query_papers

app = FastAPI(title="Paper Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class AskRequest(BaseModel):
    question: str


@app.post("/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        return run_smart_search_and_store(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = query_papers(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))  # everything is now in the SAME folder as api.py


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "h.html"))


@app.get("/c.css")
def serve_css():
    """Explicit route matching exactly what h.html asks for (href="c.css",
    no path prefix) - this is why the /static mount didn't work: it only
    answered requests to /static/c.css, not the bare /c.css the browser
    was actually asking for.
    """
    return FileResponse(os.path.join(FRONTEND_DIR, "c.css"))


@app.get("/j.js")
def serve_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "j.js"))
