const searchForm = document.getElementById("search-form");
const queryEl = document.getElementById("query");
const topKEl = document.getElementById("top-k");
const recentOnlyEl = document.getElementById("recent-only");
const urlFilterEl = document.getElementById("url-filter");
const searchBtn = document.getElementById("search-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

const dbSizeEl = document.getElementById("db-size");
const chunksSizeEl = document.getElementById("chunks-size");
const documentsCountEl = document.getElementById("documents-count");
const chunksCountEl = document.getElementById("chunks-count");
const refsCountEl = document.getElementById("refs-count");

let statsTimer = null;

function parseLineSeparated(value) {
  return value
    .split("\n")
    .map((x) => x.trim())
    .filter(Boolean);
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function runSearch(payload) {
  const response = await fetch("/api/similarity/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.message || `search failed (${response.status})`);
  }
  return data;
}

function renderResults(results) {
  if (!Array.isArray(results) || results.length === 0) {
    resultsEl.textContent = "No similarity matches.";
    return;
  }

  const lines = [];
  results.forEach((item, idx) => {
    const distance = Number(item.distance);
    const distanceLabel = Number.isFinite(distance) ? distance.toFixed(6) : "?";
    const weightedDistance = Number(item.weighted_distance);
    const weightedLabel = Number.isFinite(weightedDistance) ? weightedDistance.toFixed(6) : "?";
    const excerpt = String(item.content || "").replace(/\s+/g, " ").trim();
    const clip = excerpt.length > 360 ? `${excerpt.slice(0, 357)}...` : excerpt;

    lines.push(`[${idx + 1}] distance=${distanceLabel}`);
    lines.push(`weighted_distance: ${weightedLabel}`);
    lines.push(`citation_count: ${Number(item.citation_count || 0)}`);
    lines.push(`title: ${item.document || "(untitled)"}`);
    lines.push(`section: ${item.section || "(none)"}`);
    lines.push(`url: ${item.url || "(none)"}`);
    lines.push(`source_url: ${item.source_url || "(none)"}`);
    lines.push(`doi: ${item.doi || "(none)"}`);
    lines.push(`venue: ${item.venue || "(none)"}`);
    lines.push(`published_date: ${item.published_date || "(none)"}`);
    lines.push(`authors: ${item.authors || "(none)"}`);
    lines.push(`excerpt: ${clip || "(empty)"}`);
    lines.push("");
  });

  resultsEl.textContent = lines.join("\n");
}

async function refreshStats() {
  try {
    const response = await fetch("/api/similarity/stats", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`stats ${response.status}`);
    }
    const data = await response.json();
    const stats = data.stats || {};
    dbSizeEl.textContent = stats.database_size || "-";
    chunksSizeEl.textContent = stats.chunks_total_size || "-";
    documentsCountEl.textContent = String(stats.documents ?? 0);
    chunksCountEl.textContent = String(stats.chunks ?? 0);
    refsCountEl.textContent = String(stats.references ?? 0);
  } catch (err) {
    setStatus("Stats unavailable");
  }
}

function scheduleStatsPolling() {
  if (statsTimer !== null) {
    clearInterval(statsTimer);
  }
  statsTimer = setInterval(refreshStats, 10000);
}

searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryEl.value.trim();
  if (!query) {
    setStatus("Enter a query first.");
    return;
  }

  const payload = {
    query,
    top_k: Number.isFinite(Number(topKEl.value)) ? Number(topKEl.value) : 8,
    recent_only: Boolean(recentOnlyEl.checked),
    document_urls: parseLineSeparated(urlFilterEl.value),
  };

  searchBtn.disabled = true;
  setStatus("Searching...");
  try {
    const data = await runSearch(payload);
    renderResults(data.results || []);
    setStatus(`Returned ${Array.isArray(data.results) ? data.results.length : 0} matches.`);
  } catch (err) {
    setStatus(err.message || "Search failed.");
    resultsEl.textContent = "";
  } finally {
    searchBtn.disabled = false;
  }
});

refreshStats();
scheduleStatsPolling();
