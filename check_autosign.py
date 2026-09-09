"""
Diagnostic: trace exactly what each of the 3 credibility sources returned
for the AutoSign paper specifically, to explain the "1/3 sources found"
vs "not found in any citation database" inconsistency.

Usage:
    python check_autosign.py
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from pipe import (
    _query_semantic_scholar,
    _query_openalex,
    _query_crossref,
    get_collection,
)

TITLE = "AutoSign: Direct Pose-to-Text Translation for Continuous Sign Language Recognition"

# Pull the stored arXiv ID for AutoSign from the database, so we test with
# the EXACT id that was actually used during the real search run.
collection = get_collection()
all_data = collection.get()
arxiv_id = None
for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
    if meta["title"] == TITLE:
        # ids look like "2412.xxxxx_field" - strip the field suffix
        arxiv_id = doc_id.rsplit("_", 1)[0]
        break

print(f"Stored arXiv ID for AutoSign: {arxiv_id}\n")

print("--- Semantic Scholar ---")
s2 = _query_semantic_scholar(arxiv_id, TITLE)
print(s2)

print("\n--- OpenAlex ---")
oa = _query_openalex(arxiv_id, TITLE)
print(oa)

print("\n--- Crossref ---")
cr = _query_crossref(TITLE)
print(cr)
