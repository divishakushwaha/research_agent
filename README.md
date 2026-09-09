# Research Paper Finder_LLM Agent 

An AI-powered research assistant that takes the abstract or paper title,retrieves relevant papers from arXiv, cross-references their credibility against three independent citation databases, and produces a ranked, evidence-backed list — with an optional Q&A layer for asking follow-up questions grounded in the retrieved papers.

**Live demo:** [add your deployed URL here]

Manually reviewing literature for a research project means sifting through papers that vary wildly in relevance and credibility — some are peer-reviewed and well-cited, others are unverified preprints that happen to share keywords with your search. This tool automates that first pass: it doesn't just find papers that match a topic, it scores how trustworthy each one actually is, using real citation data instead of assumptions.

This project was built while researching literature for an Indian Sign Language (ISL) recognition project, after repeatedly running into papers that were hard to verify or only loosely relevant.

## What it does

1. **Search** — enter a topic (e.g. "sign language recognition") or a specific paper title
2. **Detection** — automatically determines whether the input is a general topic or an exact paper title, and adjusts the search strategy accordingly
3. **Relevance scoring** — an LLM scores each candidate paper's relevance to the query on a 0–10 scale, with a written justification for each score
4. **Credibility scoring** — cross-references each paper against three independent sources (Semantic Scholar, OpenAlex, Crossref) to compute a citation-based credibility score, not just raw citation count
5. **Summarization** — extracts a structured summary (problem, method, dataset, result, limitations) plus verbatim supporting quotes from each paper's abstract
6. **Q&A** — ask natural-language follow-up questions across all retrieved papers, answered using retrieval-augmented generation (RAG) grounded strictly in the stored summaries

## Architecture

```
User's browser (HTML/CSS/JS)
        │
        ▼
FastAPI backend (api.py)
        │
        ├──► arXiv API            (paper search)
        ├──► Groq API (LLM)       (relevance scoring, summarization, Q&A)
        ├──► Semantic Scholar API (citation count, venue — exact ID lookup)
        ├──► OpenAlex API         (independent citation count — cross-check)
        ├──► Crossref API         (DOI / formal publication check)
        └──► ChromaDB             (vector database — stores & retrieves summaries)
```

The frontend and backend run as a **single unified server** — FastAPI serves both the API endpoints and the static HTML/CSS/JS files from one process, so the entire app is reachable from one URL with no separate frontend server required.

---

## Technical concepts used (with plain-English definitions)

| Term | What it means here |
|---|---|
| **LLM (Large Language Model)** | The AI model (Groq's `gpt-oss-120b`) that reads paper abstracts and generates relevance scores, summaries, and answers. Called via API — no model is trained or hosted by this project. |
| **RAG (Retrieval-Augmented Generation)** | The Q&A pattern: relevant paper summaries are *retrieved* from the database first, then the LLM *generates* an answer using only that retrieved content — rather than answering from its own memory, which could be outdated or invented. |
| **Embeddings** | Numeric representations of text that capture meaning, not just keywords. Used so ChromaDB can find summaries that are *semantically* related to a question, even if they don't share exact words. |
| **Vector database** | A database (ChromaDB) optimized for storing embeddings and finding the closest matches by meaning, instead of exact-text search. |
| **Chunking** | Splitting each paper's summary into separate pieces (problem, method, dataset, result, limitations, credibility) so retrieval can match a question to the *specific* relevant field, rather than one large undifferentiated block of text. |
| **Prompt engineering** | Deliberately structuring the instructions given to the LLM (e.g. requiring exact JSON output, forcing systematic paper-by-paper checking, restricting answers to only stated facts) to make its output reliable and parseable. |
| **Citation velocity** | Citations per year since publication, rather than raw citation count — corrects for older papers appearing more credible simply because they've had more time to accumulate citations. |
| **Cross-referencing** | Querying multiple independent citation databases for the same paper and comparing results, rather than trusting a single source — flags disagreement between sources instead of silently picking one. |
| **API (Application Programming Interface)** | The mechanism by which this project's backend requests data from external services (arXiv, Groq, Semantic Scholar, OpenAlex, Crossref) over the internet. |
| **REST API endpoint** | A specific URL (e.g. `POST /search`) that accepts a request and returns a response — how the frontend and backend communicate. |
| **CORS (Cross-Origin Resource Sharing)** | A browser security mechanism that blocks a webpage from calling an API on a different origin unless explicitly permitted — configured here to allow the frontend to call the backend. |
| **Rate limiting / backoff** | Handling the case where an external API temporarily refuses requests (HTTP 429) by waiting and retrying, rather than crashing — implemented for all three credibility APIs and the Groq API. |
| **Reasoning model** | An LLM variant (tested: Qwen 3.6) that generates internal "thinking" text before its final answer — requires different output parsing than standard models. |


**Results on an 18-question evaluation set** (`openai/gpt-oss-120b`):
- Retrieval accuracy: **100%**
- Answer accuracy: **67%** (12/18)

Answer accuracy was broken down by category to identify *specific* weaknesses rather than a single opaque score:

| Category | Result |
|---|---|
| Simple fact lookup | 2/3 |
| Set completeness (listing all correct papers, not a subset) | 1/3 — weakest category |
| Negative discrimination (correctly excluding non-matches) | 3/3 |
| Cross-paper comparison | 2/2 |
| Detail precision (specific fields, not generic answers) | 0/2 — weakest category |
| Trick/robustness questions | 3/3 |
| Honest uncertainty (admitting missing data) | 1/2 |

A comparative benchmark against a reasoning model (`qwen/qwen3.6-27b`) was also run on a subset targeting the two weakest categories; it scored lower (2/5) — the reasoning step did not resolve the set-completeness or detail-precision weaknesses, and came with meaningfully higher latency and token cost. `gpt-oss-120b` was retained as the production model based on this evidence.

---

## Known limitations

- **Credibility data coverage is not universal.** Very recently published papers may not yet be indexed by Semantic Scholar or OpenAlex, even when formally published (confirmed via a real case: a 2025 ICCVW paper with a registered DOI returned zero citation-source matches). A partial mitigation credits confirmed DOI status even without citation data.
- **Credibility scores are a heuristic, not a certification.** They combine citation velocity, recency, and formal-publication signals into a transparent 0–10 score with visible reasoning — they are not a claim of peer-review quality or academic authority.
- **Answer synthesis has a measured ~67% accuracy ceiling** on multi-paper questions requiring an exhaustive list, even with full retrieval context and explicit systematic-checking instructions in the prompt.
- **Free-tier hosting** means the deployed app may take 30–60 seconds to respond after a period of inactivity (cold start).

---

## Tech stack

**Backend:** Python, FastAPI, Uvicorn
**LLM:** Groq API (`openai/gpt-oss-120b`)
**Vector database:** ChromaDB
**External data sources:** arXiv API, Semantic Scholar API, OpenAlex API, Crossref API
**Frontend:** HTML, CSS, JavaScript (vanilla — no framework)
**Deployment:** [Render / your chosen host]

---

## Running locally

```bash
# clone and enter the repo
git clone <your-repo-url>
cd <repo-folder>

# install dependencies
pip install -r requirements.txt

# set your Groq API key
export GROQ_API_KEY="your-key-here"

# run the server
uvicorn api:app --reload
```

Then open `http://localhost:8000` in your browser.

---

## Project background

Built as part of an AI engineering portfolio, originating from a real research bottleneck encountered while surveying literature for an Indian Sign Language recognition project — where distinguishing relevant, credible papers from loosely-related or unverifiable ones was a genuine, recurring time cost.
