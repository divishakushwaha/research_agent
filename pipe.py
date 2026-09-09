import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"  # silences the harmless (but noisy) telemetry errors

import sys
import json
import re
import time
import requests
import arxiv
import chromadb
from chromadb.config import Settings
from groq import Groq
LLM_MODEL = "openai/gpt-oss-120b"
DB_PATH = "paper_db"
COLLECTION_NAME = "paper_summaries"
MAX_RESULTS = 15
TOP_N_PAPERS = 10  # how many top-ranked papers to keep after scoring
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"


def safe_get(url: str, params: dict, max_retries: int = 2) -> requests.Response | None:
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                if attempt < max_retries:
                    wait = 5 * (attempt + 1)  # 5s, then 10s
                    time.sleep(wait)
                    continue
                return None
            return response
        except requests.RequestException:
            return None
    return None

RELEVANCE_GATE_PROMPT = """You are evaluating how well a paper matches a research topic, to help rank and select the best papers from a batch.

Topic being researched: {topic}

Paper title: {title}
Paper abstract: {abstract}

Score how well this paper matches the topic and explain the match to the original title/topic.
Respond with ONLY valid JSON in this exact form, no markdown fences, no preamble:
{{"relevance_score": <integer 0-10>, "match_reason": "1-2 sentences on specifically what in this paper matches (or doesn't match) the topic '{topic}'"}}
"""

SUMMARY_SCHEMA_PROMPT = """You are a research assistant extracting structured information from a paper abstract.

Given the title and abstract below, extract the following fields as JSON:
- "problem": what problem the paper addresses (1 sentence)
- "method": the core technique/approach used (1-2 sentences)
- "dataset": dataset(s) used, if mentioned, else "not specified"
- "result": key reported result or finding (1 sentence)
- "limitations": any limitations mentioned or reasonably inferred (1 sentence)
- "supporting_quotes": a list of 2-3 short EXACT sentences copied verbatim from the abstract below that best support the problem/method/result above. Do not paraphrase these - copy them exactly as written in the abstract.

Respond with ONLY valid JSON, no markdown fences, no preamble.

Title: {title}

Abstract: {abstract}
"""

ANSWER_PROMPT = """You are answering a research question using the paper summaries provided below as your only source of truth. Cite paper titles when you use information from them.

STRICT RULE: only state facts, languages, methods, or details that are explicitly written in the paper summaries below. If the summaries don't contain enough information to fully answer, say so honestly and do NOT guess, infer, or invent specifics (e.g. don't name languages, datasets, or methods that aren't in the text below, even as examples).

Some chunks below are labeled "Credibility" - these give real citation counts, venue, and a 0-10 credibility score for each paper (0 means not found in the citation database, not "bad"). Use these ONLY when the question is actually about credibility, reliability, citation counts, or which paper to trust more - do not bring them up for unrelated questions.

IMPORTANT - work through this systematically, don't skim: before writing your final answer, go through the summaries paper by paper (there may be multiple chunks per paper - treat all chunks with the same title as ONE paper) and check each one individually against the question. Do this checking step silently, then write ONLY your final answer - do not show your paper-by-paper checking in the output, just make sure every relevant paper from that check ends up mentioned in the final answer.

Keep your answer as concise as the question allows - a couple of sentences if a simple answer suffices, longer (with a table if helpful) only if the question genuinely requires comparing multiple papers in detail.

Question: {question}

Paper summaries:
{context}

Answer:
"""


def clean_json_response(raw: str) -> str:
    # reasoning on the openai model
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return raw


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key-here'"
        )
    return Groq(api_key=api_key)


def safe_groq_call(client: Groq, prompt: str, temperature: float, max_retries: int = 2) -> str:

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries:
                wait = 10 * (attempt + 1)  # 10s, then 20s
                print(f"    Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            raise  # not a rate limit, or out of retries - let it surface


def search_arxiv(topic: str, max_results: int = MAX_RESULTS):
    """Fetch candidate papers from arXiv for a given topic."""
    # delay_seconds spaces out requests to be a better citizen of arXiv's
    # API and reduce the chance of hitting rate limits (HTTP 429) after
    # repeated testing in a short window.
    client = arxiv.Client(delay_seconds=3)
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in client.results(search):
        papers.append(
            {
                "id": result.entry_id.split("/")[-1],
                "title": result.title.strip().replace("\n", " "),
                "abstract": result.summary.strip().replace("\n", " "),
                "url": result.entry_id,
                "published": str(result.published.date()),
            }
        )
    return papers


def _query_semantic_scholar(arxiv_id: str, title: str) -> dict | None:
    """Look up a paper in Semantic Scholar. Tries an EXACT match via the
    arXiv ID first (no ambiguity possible), falling back to fuzzy title
    search only if the ID lookup fails - since title text can differ in
    punctuation/subtitle formatting between arXiv and how a paper is
    indexed elsewhere, causing false "not found" results.
    """
    response = safe_get(
        f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}",
        params={"fields": "title,citationCount,venue,year"},
    )
    if response is not None and response.status_code == 200:
        match = response.json()
        return {
            "citation_count": match.get("citationCount"),
            "venue": match.get("venue") or "not specified",
            "year": match.get("year"),
            "match_method": "arxiv_id",
        }

    # --- Fallback: fuzzy title search ---
    response = safe_get(
        SEMANTIC_SCHOLAR_API,
        params={"query": title, "fields": "title,citationCount,venue,year", "limit": 1},
    )
    if response is None:
        return None
    try:
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            return None
        match = data[0]
        return {
            "citation_count": match.get("citationCount"),
            "venue": match.get("venue") or "not specified",
            "year": match.get("year"),
            "match_method": "title_search",
        }
    except (requests.RequestException, KeyError, IndexError):
        return None


def _query_openalex(arxiv_id: str, title: str) -> dict | None:
    """Look up a paper in OpenAlex via title search, verifying the match
    with a word-overlap check before trusting it. NOTE: OpenAlex does not
    support direct arXiv-ID lookup (confirmed - unlike Semantic Scholar),
    so this can't use the exact-ID approach; the overlap check is the
    safeguard against title search returning a wrong paper.
    """
    response = safe_get(OPENALEX_API, params={"search": title, "per_page": 3})
    if response is None:
        return None
    try:
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None

        # Only accept a match if its title is genuinely close to ours -
        # simple word-overlap check, enough to reject an obviously wrong
        # result without needing a full fuzzy-match library.
        our_words = set(title.lower().split())
        best_match, best_overlap = None, 0.0
        for candidate in results:
            cand_words = set((candidate.get("title") or "").lower().split())
            if not our_words:
                continue
            overlap = len(our_words & cand_words) / len(our_words)
            if overlap > best_overlap:
                best_match, best_overlap = candidate, overlap

        if best_match is None or best_overlap < 0.6:
            return None  # too risky to trust - treat as not found

        return {
            "citation_count": best_match.get("cited_by_count"),
            "venue": (best_match.get("primary_location") or {}).get("source", {}).get("display_name")
            if best_match.get("primary_location") else None,
            "year": best_match.get("publication_year"),
            "match_method": f"title_search ({best_overlap:.0%} word overlap)",
        }
    except (requests.RequestException, KeyError, IndexError, AttributeError):
        return None


def _query_crossref(title: str) -> dict | None:
    """Look up a paper by title in Crossref - the official DOI registry.
    Used specifically to check whether a paper has a real DOI, meaning it
    was FORMALLY published (journal/conference), not just an arXiv
    preprint. This is a genuinely different signal than citation count.
    """
    response = safe_get(CROSSREF_API, params={"query.bibliographic": title, "rows": 1})
    if response is None:
        return None
    try:
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        if not items:
            return None
        match = items[0]
        return {
            "has_doi": bool(match.get("DOI")),
            "publisher": match.get("publisher"),
            "container_title": (match.get("container-title") or [None])[0],
        }
    except (requests.RequestException, KeyError, IndexError):
        return None


def get_credibility_info(paper: dict) -> dict:
    """Look up REAL, verifiable credibility signals for a paper by
    cross-referencing THREE independent, free, no-key-required sources:
    Semantic Scholar, OpenAlex (independent citation count), and Crossref
    (formal-publication / DOI check). None of this is an LLM guess -
    these are facts pulled from actual citation databases.

    Cross-referencing matters because any single database can be wrong
    or missing data for a given paper - agreement across sources is
    itself a signal, and using two independently-computed citation
    counts lets us sanity-check rather than blindly trust one number.
    """
    title = paper["title"]
    # paper["id"] looks like "2401.12345v2" - strip the version suffix
    # since arXiv IDs in other databases are typically unversioned.
    arxiv_id = re.sub(r"v\d+$", "", paper["id"])

    s2 = _query_semantic_scholar(arxiv_id, title)
    oa = _query_openalex(arxiv_id, title)
    cr = _query_crossref(title)

    sources_found = sum(x is not None for x in [s2, oa, cr])

    # Take whichever citation counts are available (could be 0, 1, or 2)
    citation_counts = [
        x["citation_count"] for x in [s2, oa]
        if x is not None and x.get("citation_count") is not None
    ]

    # Use the average of available counts (not just picking one source
    # arbitrarily) as the working citation count for scoring.
    citation_count = round(sum(citation_counts) / len(citation_counts)) if citation_counts else None

    # Flag if the two sources disagree substantially (>2x apart) - this
    # is shown to the user rather than silently hidden, since a big
    # disagreement means the "true" citation count is genuinely uncertain.
    citation_disagreement = None
    if len(citation_counts) == 2 and max(citation_counts) > 0:
        ratio = max(citation_counts) / max(min(citation_counts), 1)
        citation_disagreement = ratio > 2.0

    year = (s2 or {}).get("year") or (oa or {}).get("year")
    venue = (s2 or {}).get("venue") or (oa or {}).get("venue") or "not specified"
    has_doi = (cr or {}).get("has_doi", False)

    return {
        "citation_count": citation_count,
        "citation_counts_by_source": {
            "semantic_scholar": (s2 or {}).get("citation_count"),
            "openalex": (oa or {}).get("citation_count"),
        },
        "match_methods": {
            "semantic_scholar": (s2 or {}).get("match_method"),
            "openalex": (oa or {}).get("match_method"),
        },
        "citation_disagreement": citation_disagreement,
        "year": year,
        "venue": venue,
        "has_doi": has_doi,
        "sources_found": sources_found,  # 0-3, how many of the 3 APIs found this paper at all
    }


def score_credibility(info: dict) -> dict:
    """Combine citation velocity, recency, formal-publication status, and
    cross-source agreement into one 0-10 credibility score - with the
    REASONING kept visible, not collapsed into an opaque number.

    This is a transparent heuristic, not a claim of true academic impact.
    It's built from real signals to reduce (not eliminate) two known
    biases: citation count favoring old papers, and single-source data
    being incomplete or wrong.
    """
    import datetime
    current_year = datetime.datetime.now().year
    reasons = []

    if info["citation_count"] is None or info["sources_found"] == 0:
        # Even with no citation count, a confirmed DOI (formal publication)
        # is a real, independent credibility signal - don't treat this
        # the same as "found nowhere at all". Common for very recent
        # papers where Semantic Scholar/OpenAlex haven't indexed yet but
        # Crossref already has the DOI registered.
        if info["has_doi"]:
            return {
                "score": 3,
                "reasons": [
                    "Not yet indexed by citation-count sources (likely very recent), but confirmed formally published via a registered DOI (Crossref) - some credibility signal despite no citation data yet."],
            }
        return {
            "score": 0,
            "reasons": [
                "Not found in any citation database (Semantic Scholar, OpenAlex, or Crossref) - may be very new, very niche, or a title-match failure."],
        }

    count = info["citation_count"]
    year = info["year"]

    # --- Citation velocity: citations per year, not raw count ---
    # This directly counters "recent papers look worse" - a 2-year-old
    # paper with 20 citations (10/year) is doing better than a 10-year-old
    # paper with 40 citations (4/year), even though 40 > 20 in raw count.
    if year:
        age = max(current_year - year, 1)  # avoid divide-by-zero for this-year papers
        velocity = count / age
    else:
        velocity = count  # no year data - fall back to raw count
        age = None

    if velocity >= 20:
        base = 10
    elif velocity >= 10:
        base = 8
    elif velocity >= 4:
        base = 6
    elif velocity >= 1:
        base = 4
    elif velocity > 0:
        base = 2
    else:
        base = 1
    reasons.append(f"Citation velocity: {velocity:.1f}/year ({count} citations over {age or '?'} years) -> base score {base}")

    # --- Recency bonus: offset natural citation lag for very new papers ---
    if year and (current_year - year) <= 2:
        base = min(base + 2, 10)
        reasons.append(f"Recency bonus (+2, capped at 10): published within the last 2 years, citations haven't caught up yet")

    # --- Formal publication bonus: has a real DOI (journal/conference), not just a preprint ---
    if info["has_doi"]:
        base = min(base + 1, 10)
        reasons.append("Formal publication bonus (+1): has a registered DOI (Crossref), indicating formal journal/conference publication rather than preprint-only")

    # --- Cross-source agreement note (informational, not scored) ---
    if info["citation_disagreement"] is True:
        reasons.append(
            f"CAUTION: citation counts disagree across sources ({info['citation_counts_by_source']}) - treat this score as less certain."
        )
    elif info["sources_found"] >= 2:
        reasons.append(f"Confirmed by {info['sources_found']}/3 sources - citation count is cross-verified.")

    return {"score": base, "reasons": reasons}


def score_relevance(client: Groq, topic: str, paper: dict) -> dict:
    """Ask the LLM to SCORE (not just gate) how well this paper matches the
    topic, and explain the specific match. Used to rank a whole batch of
    papers and keep only the top N, rather than a simple yes/no filter.
    """
    prompt = RELEVANCE_GATE_PROMPT.format(
        topic=topic, title=paper["title"], abstract=paper["abstract"]
    )
    content = safe_groq_call(client, prompt, temperature=0.0)
    raw = clean_json_response(content)
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        # If scoring fails to parse, give it a middling score rather than
        # silently dropping it - fail open, not closed.
        verdict = {"relevance_score": 5, "match_reason": "relevance scoring failed to parse"}
    return verdict


def summarize_paper(client: Groq, paper: dict) -> dict:
    """Call the LLM to extract a structured summary from one paper's abstract."""
    prompt = SUMMARY_SCHEMA_PROMPT.format(
        title=paper["title"], abstract=paper["abstract"]
    )
    content = safe_groq_call(client, prompt, temperature=0.2)
    raw = clean_json_response(content)
    try:
        structured = json.loads(raw)
    except json.JSONDecodeError:
        structured = {
            "problem": "parse_error",
            "method": "parse_error",
            "dataset": "parse_error",
            "result": "parse_error",
            "limitations": "parse_error",
        }
    return structured


def get_collection():
    db_client = chromadb.PersistentClient(
        path=DB_PATH, settings=Settings(anonymized_telemetry=False)
    )
    return db_client.get_or_create_collection(name=COLLECTION_NAME)


def store_summaries(papers_with_summaries: list):
    """Store each paper's structured summary in ChromaDB, one chunk per field.

    At small scale (a handful of papers), one big text block per paper works
    fine. At larger scale, splitting each field (problem/method/dataset/
    result/limitations) into its OWN embedded chunk makes retrieval more
    precise - a question specifically about "dataset" matches the dataset
    chunk directly, instead of competing with unrelated text in the same
    block. All chunks from one paper share the same metadata so results can
    still be traced back to their source paper.
    """
    collection = get_collection()

    ids, documents, metadatas = [], [], []
    for paper in papers_with_summaries:
        s = paper["summary"]
        # ChromaDB metadata only accepts str/int/float/bool - None is
        # rejected outright. citation_count and venue can legitimately be
        # None when a paper isn't found in any credibility source, so we
        # convert those to explicit sentinel values here rather than
        # passing None straight through.
        base_meta = {
            "title": paper["title"],
            "url": paper["url"],
            "published": paper["published"],
            "relevance_score": paper.get("relevance_score") or 0,
            "citation_count": paper.get("citation_count") if paper.get("citation_count") is not None else -1,
            "credibility_score": paper.get("credibility_score") or 0,
            "venue": paper.get("venue") or "not specified",
        }
        for field in ["problem", "method", "dataset", "result", "limitations"]:
            ids.append(f"{paper['id']}_{field}")
            documents.append(f"[{paper['title']}] {field.capitalize()}: {s[field]}")
            metadatas.append({**base_meta, "field": field})

        # Store the match reason as its own chunk too, so "why is this paper
        # relevant" is itself something you can query for later.
        if "match_reason" in paper:
            ids.append(f"{paper['id']}_match_reason")
            documents.append(f"[{paper['title']}] Match to search topic: {paper['match_reason']}")
            metadatas.append({**base_meta, "field": "match_reason"})

        # Store credibility as its own queryable chunk too - so "how
        # credible is X" or "which papers are most cited" can be answered
        # from stored text, same as any other field. Reasons are included
        # so the SCORE isn't a black box - the "why" travels with it.
        if paper.get("citation_count") is not None:
            reasons_text = " ".join(paper.get("credibility_reasons", []))
            cred_text = (
                f"[{paper['title']}] Credibility: {paper['citation_count']} citations "
                f"(cross-referenced across {paper.get('sources_found', 0)}/3 sources), "
                f"published in {paper.get('venue', 'not specified')}, "
                f"credibility score {paper.get('credibility_score')}/10. Reasoning: {reasons_text}"
            )
        else:
            cred_text = f"[{paper['title']}] Credibility: not found in any of the 3 citation databases checked - citation data unavailable."
        ids.append(f"{paper['id']}_credibility")
        documents.append(cred_text)
        metadatas.append({**base_meta, "field": "credibility"})

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def run_search_and_store(topic: str):
    client = get_client()
    print(f"Searching arXiv for: {topic}")
    papers = search_arxiv(topic)
    print(f"Found {len(papers)} papers. Scoring relevance to '{topic}'...")

    # Score EVERY candidate paper first (cheap - just relevance_score + reason,
    # no full extraction yet), then rank and keep only the best matches.
    scored = []
    for i, paper in enumerate(papers, 1):
        print(f"  [{i}/{len(papers)}] Scoring: {paper['title'][:60]}...")
        verdict = score_relevance(client, topic, paper)
        paper["relevance_score"] = verdict["relevance_score"]
        paper["match_reason"] = verdict["match_reason"]
        scored.append(paper)

    # Rank by score, descending, and take the top N (default 10).
    scored.sort(key=lambda p: p["relevance_score"], reverse=True)
    top_papers = scored[:TOP_N_PAPERS]

    print(f"\nTop {len(top_papers)} matches for '{topic}':")
    for p in top_papers:
        print(f"  [{p['relevance_score']}/10] {p['title'][:60]}...")
        print(f"           {p['match_reason']}")

    # Look up REAL credibility signals via THREE independent sources
    # (Semantic Scholar, OpenAlex, Crossref) - only for the top N papers
    # that made the cut, not all candidates, to keep this fast.
    print(f"\nLooking up credibility info (cross-referencing 3 sources)...")
    for i, paper in enumerate(top_papers, 1):
        info = get_credibility_info(paper)
        result = score_credibility(info)
        paper["citation_count"] = info["citation_count"]
        paper["venue"] = info["venue"]
        paper["credibility_score"] = result["score"]
        paper["credibility_reasons"] = result["reasons"]
        paper["sources_found"] = info["sources_found"]
        print(f"  [{i}/{len(top_papers)}] {paper['title'][:50]}... -> {result['score']}/10 ({info['sources_found']}/3 sources)")
        time.sleep(1)  # small pacing delay - be a good citizen of these free APIs

    # Only now do full structured extraction - on the top N, not every result.
    results = []
    for paper in top_papers:
        summary = summarize_paper(client, paper)
        paper["summary"] = summary
        results.append(paper)
        time.sleep(1)  # small pacing delay between Groq calls

    count = store_summaries(results)
    print(f"\nStored {count} paper summaries in {DB_PATH}")

    # ---- THE ACTUAL PRODUCT OUTPUT: ranked list by credibility ----
    print(f"\n{'=' * 60}")
    print(f"CREDIBLE PAPERS ON '{topic}' (ranked by credibility score)")
    print(f"{'=' * 60}")
    ranked = sorted(results, key=lambda p: p["credibility_score"], reverse=True)
    for rank, p in enumerate(ranked, 1):
        print(f"\n{rank}. [{p['credibility_score']}/10] {p['title']}")
        print(f"   Relevance: {p['relevance_score']}/10 - {p['match_reason']}")
        if p["citation_count"] is not None:
            print(f"   Citations: {p['citation_count']} (verified across {p['sources_found']}/3 sources) | Venue: {p['venue']}")
        else:
            print(f"   Citations: not found in any citation database")
        print(f"   Summary: {p['summary']['problem']}")

    return results
# ============================================================
# ADD THIS TO paper_agent.py (new function + one pipeline change)
# ============================================================

def search_arxiv_exact_title(title_query: str, max_results: int = 3):
    """Try to find an EXACT paper by title using arXiv's title-field
    search (ti:"..."). Returns a list of 0-3 candidate exact matches -
    empty list means no exact title match was found, so the caller
    should fall back to treating the input as a general topic instead.
    """
    client = arxiv.Client(delay_seconds=3)
    # ti:"..." searches ONLY the title field for that exact phrase -
    # very different from a broad keyword search across title+abstract.
    search = arxiv.Search(
        query=f'ti:"{title_query}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "id": result.entry_id.split("/")[-1],
                "title": result.title.strip().replace("\n", " "),
                "abstract": result.summary.strip().replace("\n", " "),
                "url": result.entry_id,
                "published": str(result.published.date()),
            })
    except Exception:
        return []  # if the exact-title query itself errors, fall back to topic search
    return papers


def is_likely_paper_title(query: str, exact_matches: list) -> bool:
    """Decide whether the user's input should be TREATED as a specific
    paper name (show it as the primary result) vs a general topic.

    We don't guess based on text length/capitalization - that's fragile.
    Instead we trust arXiv itself: if the exact title-field search found
    something reasonably close, treat this as a paper lookup. This means
    the decision is grounded in a real match, not a guess about the
    user's intent from the text alone.
    """
    if not exact_matches:
        return False
    # Require the matched title to substantially overlap with what the
    # user typed - guards against ti:"..." matching on a loose partial
    # phrase for what was actually meant as a topic.
    query_words = set(query.lower().split())
    top_match_words = set(exact_matches[0]["title"].lower().split())
    if not query_words:
        return False
    overlap = len(query_words & top_match_words) / len(query_words)
    return overlap >= 0.7  # most of what they typed appears in the title


def run_smart_search_and_store(user_input: str):
    """The new entry point for the app: decides whether user_input is a
    specific paper name or a general topic, and returns results shaped
    accordingly - EITHER (exact_paper, similar_papers) OR (None, topic_papers).
    """
    print(f"Checking if '{user_input}' is an exact paper title...")
    exact_candidates = search_arxiv_exact_title(user_input)

    if is_likely_paper_title(user_input, exact_candidates):
        exact_title = exact_candidates[0]["title"]
        print(f"Found exact match: {exact_title}")

        # Run the EXACT paper through the full pipeline (credibility,
        # summary) - reuse existing functions, no duplicated logic.
        client = get_client()
        exact_paper = exact_candidates[0]
        verdict = score_relevance(client, user_input, exact_paper)
        exact_paper["relevance_score"] = verdict["relevance_score"]
        exact_paper["match_reason"] = "Exact title match for your search."
        info = get_credibility_info(exact_paper)
        result = score_credibility(info)
        exact_paper["citation_count"] = info["citation_count"]
        exact_paper["venue"] = info["venue"]
        exact_paper["credibility_score"] = result["score"]
        exact_paper["credibility_reasons"] = result["reasons"]
        exact_paper["sources_found"] = info["sources_found"]
        exact_paper["summary"] = summarize_paper(client, exact_paper)

        # Now get SIMILAR papers using the normal broad topic search,
        # for the "similar papers" section underneath.
        print("Finding similar papers...")
        similar_results = run_search_and_store(exact_title)
        # Don't show the exact paper twice if it also shows up in the
        # similar-papers list.
        similar_results = [p for p in similar_results if p["id"] != exact_paper["id"]]

        store_summaries([exact_paper])  # store the exact paper too
        return {"exact_paper": exact_paper, "similar_papers": similar_results}

    else:
        print(f"No exact title match - treating '{user_input}' as a topic search.")
        topic_results = run_search_and_store(user_input)
        return {"exact_paper": None, "similar_papers": topic_results}



def query_papers(question: str, n_results: int = 30) -> str:
    """Semantic search over stored summaries, then synthesize an answer.

    Each paper is stored as 6 field-level chunks (5 fields + match_reason).
    At small scale (a few dozen papers), semantic search can still miss a
    relevant chunk due to imperfect embedding matches - so below a size
    threshold, we just retrieve the ENTIRE collection instead of doing a
    partial similarity search. This trades a slightly longer prompt for
    guaranteed completeness, which is the right trade at this scale. Above
    the threshold, the same partial retrieval as before still applies,
    since sending the whole database in every prompt stops being practical.
    """
    collection = get_collection()
    available = collection.count()

    FULL_RETRIEVAL_THRESHOLD = 100  # chunks - well beyond a ~15-paper database

    if available <= FULL_RETRIEVAL_THRESHOLD:
        # Retrieve everything - "n_results" larger than the collection just
        # returns the whole thing.
        hits = collection.query(query_texts=[question], n_results=available)
    else:
        hits = collection.query(query_texts=[question], n_results=n_results)

    documents = hits["documents"][0]
    metadatas = hits["metadatas"][0]

    if not documents:
        return "No papers found in the database yet. Run a search first."

    # Group chunks by paper title so each paper appears as ONE clearly
    # separated block, instead of its fields being scattered throughout
    # the prompt in retrieval-score order. This makes it much easier for
    # the LLM to treat "all chunks with this title" as a single paper.
    papers = {}
    for doc, meta in zip(documents, metadatas):
        title = meta["title"]
        papers.setdefault(title, []).append(doc)

    context = "\n\n".join(
        f"=== PAPER: {title} ===\n" + "\n".join(chunks)
        for title, chunks in papers.items()
    )

    client = get_client()
    content = safe_groq_call(
        client, ANSWER_PROMPT.format(question=question, context=context), temperature=0.3
    )
    # This is a free-text answer, not JSON - reasoning models may still
    # wrap it in <think> tags, so strip those, but don't attempt JSON parse.
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print('  python paper_agent.py search "<topic>"')
        print('  python paper_agent.py ask "<question>"')
        sys.exit(1)

    command, arg = sys.argv[1], sys.argv[2]

    if command == "search":
        run_search_and_store(arg)
    elif command == "ask":
        answer = query_papers(arg)
        print("\n" + answer)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)