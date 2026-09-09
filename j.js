const API_BASE = "";

const searchInput = document.getElementById("search-input");
const searchButton = document.getElementById("search-button");
const statusMessage = document.getElementById("status-message");
const exactMatchSection = document.getElementById("exact-match-section");
const exactMatchCard = document.getElementById("exact-match-card");
const resultsSection = document.getElementById("results-section");
const resultsHeading = document.getElementById("results-heading");
const resultsList = document.getElementById("results-list");
const qaSection = document.getElementById("qa-section");
const questionInput = document.getElementById("question-input");
const askButton = document.getElementById("ask-button");
const answerBox = document.getElementById("answer-box");

function credibilityBadge(score) {
    if (score >= 7) return { label: "High Credibility", color: "green" };
    if (score >= 4) return { label: "Moderate Credibility", color: "orange" };
    return { label: "Low Credibility / Unverified", color: "red" };
}

function renderPaperCard(paper, rank) {
    const badge = credibilityBadge(paper.credibility_score);
    const rankPrefix = rank ? `${rank}. ` : "";

    const citationsText = paper.citation_count !== null && paper.citation_count !== undefined
        ? `📊 ${paper.citation_count} citations (${paper.sources_found}/3 sources)`
        : "📊 Citation data unavailable";

    const quotes = (paper.summary.supporting_quotes || [])
        .map(q => `<blockquote>"${escapeHtml(q)}"</blockquote>`)
        .join("");

    const reasons = (paper.credibility_reasons || [])
        .map(r => `<div>. ${escapeHtml(r)}</div>`)
        .join("");

    return `
        <div class="paper-card">
            <div class="paper-title">${rankPrefix}${escapeHtml(paper.title)}</div>
            <span class="badge ${badge.color}">${badge.label} (${paper.credibility_score}/10)</span>
            <div class="paper-facts">
                <a href="${paper.url}" target="_blank">View paper</a>
                <span>${escapeHtml(paper.venue || "not specified")}</span>
                <span>${citationsText}</span>
            </div>
            <details>
                <summary>See details (abstract summary, supporting quotes, credibility reasoning)</summary>
                <div class="detail-content">
                    <p><strong>Why this matched:</strong> ${escapeHtml(paper.match_reason)} <em>(relevance ${paper.relevance_score}/10)</em></p>
                    <p><strong>Problem:</strong> ${escapeHtml(paper.summary.problem)}</p>
                    <p><strong>Method:</strong> ${escapeHtml(paper.summary.method)}</p>
                    <p><strong>Dataset:</strong> ${escapeHtml(paper.summary.dataset)}</p>
                    <p><strong>Result:</strong> ${escapeHtml(paper.summary.result)}</p>
                    <p><strong>Limitations:</strong> ${escapeHtml(paper.summary.limitations)}</p>
                    ${quotes ? `<p><strong>Supporting quotes:</strong></p>${quotes}` : ""}
                    ${reasons ? `<div class="credibility-reasons"><strong>Credibility reasoning:</strong>${reasons}</div>` : ""}
                </div>
            </details>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = type;
}

function clearStatus() {
    statusMessage.textContent = "";
    statusMessage.className = "";
}

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    searchButton.disabled = true;
    exactMatchSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
    qaSection.classList.add("hidden");
    showStatus(`Searching for "${query}"... this takes a few minutes.`, "loading");

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Search failed.");
        }

        const result = await response.json();
        clearStatus();
        displayResults(result);
    } catch (err) {
        showStatus(`Error: ${err.message}`, "error");
    } finally {
        searchButton.disabled = false;
    }
}

function displayResults(result) {
    if (result.exact_paper) {
        exactMatchCard.innerHTML = renderPaperCard(result.exact_paper, null);
        exactMatchSection.classList.remove("hidden");

        if (result.similar_papers && result.similar_papers.length > 0) {
            const ranked = [...result.similar_papers].sort((a, b) => b.credibility_score - a.credibility_score);
            resultsHeading.textContent = `Similar papers (${ranked.length})`;
            resultsList.innerHTML = ranked.map((p, i) => renderPaperCard(p, i + 1)).join("");
            resultsSection.classList.remove("hidden");
        }
    } else {
        const papers = result.similar_papers || [];
        const ranked = [...papers].sort((a, b) => b.credibility_score - a.credibility_score);
        resultsHeading.textContent = `Results (${ranked.length} papers)`;
        resultsList.innerHTML = ranked.map((p, i) => renderPaperCard(p, i + 1)).join("");
        resultsSection.classList.remove("hidden");
    }

    qaSection.classList.remove("hidden");
}

async function performAsk() {
    const question = questionInput.value.trim();
    if (!question) return;

    askButton.disabled = true;
    answerBox.textContent = "Thinking...";

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Question failed.");
        }

        const result = await response.json();
        answerBox.textContent = result.answer;
    } catch (err) {
        answerBox.textContent = `Error: ${err.message}`;
    } finally {
        askButton.disabled = false;
    }
}

searchButton.addEventListener("click", performSearch);
searchInput.addEventListener("keydown", (e) => { if (e.key === "Enter") performSearch(); });
askButton.addEventListener("click", performAsk);
questionInput.addEventListener("keydown", (e) => { if (e.key === "Enter") performAsk(); });
