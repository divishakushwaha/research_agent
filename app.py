import streamlit as st
from pipe import run_smart_search_and_store, query_papers

st.set_page_config(page_title="Paper Research Agent", page_icon="📄", layout="wide")

st.title("📄 Paper Research Agent")
st.caption("Enter a topic or a specific paper title - we'll detect which one it is.")

if "search_result" not in st.session_state:
    st.session_state.search_result = None


def credibility_badge(score: int) -> tuple[str, str]:
    """Turn a 0-10 credibility score into (label, color) for a real
    colored badge - not an emoji. Streamlit's st.badge() accepts a small
    fixed set of color names, which is what we map to here. Thresholds
    are fixed and consistent across every paper: green = well-established,
    yellow (Streamlit calls it "orange") = moderate/uncertain, red =
    unverified or very new/uncited.
    """
    if score >= 7:
        return "High Credibility", "green"
    elif score >= 4:
        return "Moderate Credibility", "orange"
    else:
        return "Low Credibility / Unverified", "red"

user_input = st.text_input(
    "Topic or paper title",
    placeholder="e.g. sign language recognition OR a full paper title",
)
search_clicked = st.button("Search", type="primary")

if search_clicked and user_input:
    with st.spinner(f"Searching for '{user_input}'... this takes a few minutes."):
        st.session_state.search_result = run_smart_search_and_store(user_input)


def render_paper_card(paper: dict, rank: int | None = None):
    """Render one paper's card - title, credibility badge, quick facts row,
    and an expander with full details. Used for both the exact-match card
    and every card in the similar-papers list, so the two sections look
    consistent.
    """
    label, color = credibility_badge(paper["credibility_score"])
    heading = f"{rank}. {paper['title']}" if rank else paper["title"]
    st.markdown(f"### {heading}")
    st.badge(f"{label} ({paper['credibility_score']}/10)", color=color)

    col1, col2, col3 = st.columns(3)
    col1.write(f"🔗 [View paper]({paper['url']})")
    col2.write(f"🏛️ {paper.get('venue', 'not specified')}")
    if paper["citation_count"] is not None:
        col3.write(f"📊 {paper['citation_count']} citations ({paper['sources_found']}/3 sources)")
    else:
        col3.write("📊 Citation data unavailable")

    with st.expander("See details (abstract summary, supporting quotes, credibility reasoning)"):
        st.markdown(f"**Why this matched:** {paper['match_reason']}  \n*(relevance {paper['relevance_score']}/10)*")

        st.markdown("**Abstract summary:**")
        st.write(f"- **Problem:** {paper['summary']['problem']}")
        st.write(f"- **Method:** {paper['summary']['method']}")
        st.write(f"- **Dataset:** {paper['summary']['dataset']}")
        st.write(f"- **Result:** {paper['summary']['result']}")
        st.write(f"- **Limitations:** {paper['summary']['limitations']}")

        quotes = paper["summary"].get("supporting_quotes", [])
        if quotes:
            st.markdown(f"**Supporting quotes ({len(quotes)}):**")
            for quote in quotes:
                st.markdown(f"> \"{quote}\"")

        st.markdown("**Credibility reasoning:**")
        for reason in paper.get("credibility_reasons", []):
            st.write(f"- {reason}")

    st.divider()

if st.session_state.search_result:
    result = st.session_state.search_result

    if result["exact_paper"]:
        st.subheader("✅ Exact match found")
        render_paper_card(result["exact_paper"])

        if result["similar_papers"]:
            st.subheader(f"Similar papers ({len(result['similar_papers'])})")
            ranked = sorted(result["similar_papers"], key=lambda p: p["credibility_score"], reverse=True)
            for rank, paper in enumerate(ranked, 1):
                render_paper_card(paper, rank)
    else:
        # No exact match - this was a topic search, show the normal
        # keyword-matched, credibility-ranked list.
        papers = result["similar_papers"]
        st.subheader(f"Results ({len(papers)} papers)")
        ranked = sorted(papers, key=lambda p: p["credibility_score"], reverse=True)
        for rank, paper in enumerate(ranked, 1):
            render_paper_card(paper, rank)
    st.subheader("Ask a question about these papers")
    question = st.text_input("Your question", placeholder="e.g. which papers used pose estimation?")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer = query_papers(question)
        st.write(answer)
