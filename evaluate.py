"""
Evaluation script for the paper research agent.

Story: this is a spot-check exam you write for your own system. You define
questions where YOU already know a correct answer (because you've read
these papers), run the pipeline, and score how well it did. This turns
"it seems to work" into an actual measured number.

Two things get evaluated separately, because they can fail independently:

1. RETRIEVAL quality - did ChromaDB pull back chunks from the RIGHT papers
   for a given question? (measured with simple keyword/title matching)
2. ANSWER quality - is the LLM's final answer actually correct, given what
   was retrieved? (measured by asking a second LLM call to grade it - this
   is called "LLM-as-judge", a common real-world evaluation technique)

Usage:
    python evaluate.py
"""

import json
import re
import time
from pipe import get_client, query_papers, get_collection, LLM_MODEL
TEST_SET =[
    {
        "question": "Which papers used pose estimation?",
        "expected_paper_keywords": ["AutoSign", "In-Car Sign Language Corpus", "ChaLearn"],
        "expected_answer_summary": "Should identify AutoSign, the ICSL corpus, and ChaLearn LAP as using pose/motion-capture data - all three, not a subset.",
    },
    {
        "question": "Which papers use skeleton or pose data instead of raw video?",
        "expected_paper_keywords": ["Skeleton Aware", "AutoSign", "In-Car Sign Language Corpus"],
        "expected_answer_summary": "Should identify Skeleton Aware Multi-modal SLR, AutoSign, and the ICSL corpus - all three papers, not a partial list.",
    },
    {
        "question": "Which papers focus on Russian Sign Language specifically?",
        "expected_paper_keywords": ["Slovo", "Bukva"],
        "expected_answer_summary": "Should identify BOTH Slovo and Bukva as Russian Sign Language resources.",
    },

    # ---- Detail precision (gpt-oss-120b's other weak category: 0/2) ----
    {
        "question": "What limitation did the paper on isolated sign language training strategies report?",
        "expected_paper_keywords": ["Training Strategies"],
        "expected_answer_summary": "Should pull the actual limitation text stored for that paper's summary, not a generic guess about SLR limitations.",
    },
    {
        "question": "What method does the multimodal pre-training paper for sign language understanding use?",
        "expected_paper_keywords": ["Scaling up Multimodal"],
        "expected_answer_summary": "Should describe the actual method from that paper's stored summary, not confuse it with a different multimodal paper like ICSL or Skeleton Aware.",
    },
]


def check_retrieval(question: str, expected_keywords: list, n_results: int = 30):
    """Check whether the retrieved chunks come from the expected papers."""
    collection = get_collection()
    available = collection.count()
    effective_n = available if available <= 100 else min(n_results, available)
    hits = collection.query(query_texts=[question], n_results=effective_n)
    retrieved_titles = {m["title"] for m in hits["metadatas"][0]}

    matched = [
        kw for kw in expected_keywords
        if any(kw.lower() in title.lower() for title in retrieved_titles)
    ]
    return {
        "expected": expected_keywords,
        "matched": matched,
        "retrieved_titles": list(retrieved_titles),
        "score": len(matched) / len(expected_keywords) if expected_keywords else None,
    }


def grade_answer(question: str, actual_answer: str, expected_summary: str) -> dict:
    """Use the LLM itself as a judge: does the actual answer match what a
    correct answer should contain? This is imperfect (the judge can be
    wrong too) but far better than no measurement at all, and it's the
    standard approach used in real RAG evaluation.
    """
    client = get_client()
    grading_prompt = f"""You are grading whether an AI's answer to a research question is correct.

Question: {question}

What a correct answer should cover: {expected_summary}

The AI's actual answer: {actual_answer}

Grade this as JSON only, no markdown fences:
{{"correct": true or false, "reasoning": "one sentence explaining the grade"}}
"""
    # Retry on rate limits instead of crashing the whole evaluation run -
    # a single transient 429 shouldn't lose all progress on the other
    # 17 questions.
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": grading_prompt}],
                temperature=0.0,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"    Rate limited, waiting 15s before retry...")
                time.sleep(15)
            else:
                raise
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models (Qwen included) sometimes wrap the JSON in extra
        # reasoning text even when told not to. As a fallback, try to
        # find just the {...} block anywhere in the response before
        # giving up entirely.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"correct": None, "reasoning": f"grading response failed to parse: {raw[:150]}"}


def run_evaluation():
    if not TEST_SET:
        print("TEST_SET is empty - add questions with known answers before running.")
        return

    retrieval_scores = []
    answer_results = []

    for i, case in enumerate(TEST_SET, 1):
        print(f"\n[{i}/{len(TEST_SET)}] {case['question']}")

        retrieval = check_retrieval(case["question"], case["expected_paper_keywords"])
        retrieval_scores.append(retrieval["score"])
        print(f"  Retrieval: {len(retrieval['matched'])}/{len(retrieval['expected'])} expected papers found")
        if retrieval["score"] is not None and retrieval["score"] < 1.0:
            missing = set(retrieval["expected"]) - set(retrieval["matched"])
            print(f"  Missing: {missing}")

        answer = query_papers(case["question"])
        grade = grade_answer(case["question"], answer, case["expected_answer_summary"])
        answer_results.append(grade["correct"])
        print(f"  Answer correct: {grade['correct']} - {grade['reasoning']}")
        time.sleep(8)
    valid_retrieval = [s for s in retrieval_scores if s is not None]
    avg_retrieval = sum(valid_retrieval) / len(valid_retrieval) if valid_retrieval else 0
    correct_count = sum(1 for r in answer_results if r is True)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Average retrieval score: {avg_retrieval:.0%}")
    print(f"Answer accuracy: {correct_count}/{len(TEST_SET)} ({correct_count / len(TEST_SET):.0%})")


if __name__ == "__main__":
    run_evaluation()