from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import uuid

from pipe import run_smart_search_and_store, query_papers

app = FastAPI(title="Paper Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],)
jobs = {}


class SearchRequest(BaseModel):
    query: str


class AskRequest(BaseModel):
    question: str


def run_search_job(job_id: str, query: str):
    """This runs in the background, AFTER the /search endpoint has
    already responded to the browser. It does the actual slow work.
    """
    try:
        result = run_smart_search_and_store(query)
        jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        jobs[job_id] = {"status": "error", "detail": str(e)}


@app.post("/search")
def start_search(request: SearchRequest, background_tasks: BackgroundTasks):
    """Starts a search WITHOUT waiting for it to finish. Returns a job_id
    immediately - the actual pipeline runs in the background via
    background_tasks.add_task(), which FastAPI executes after this
    function returns its response.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running"}
    background_tasks.add_task(run_search_job, job_id, request.query)
    return {"job_id": job_id}


@app.get("/search-status/{job_id}")
def get_search_status(job_id: str):
    """The frontend calls this repeatedly to check whether a job is done."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


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

