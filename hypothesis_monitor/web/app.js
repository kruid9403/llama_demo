const form = document.getElementById("hypothesis-form");
const promptEl = document.getElementById("prompt");
const topKEl = document.getElementById("top-k");
const recentOnlyEl = document.getElementById("recent-only");
const urlFilterEl = document.getElementById("url-filter");
const runBtn = document.getElementById("run-btn");
const statusEl = document.getElementById("status");
const progressFillEl = document.getElementById("progress-fill");
const progressLabelEl = document.getElementById("progress-label");
const progressWrapEl = document.querySelector(".progress-wrap");

const queryOutputEl = document.getElementById("query-output");
const reviewOutputEl = document.getElementById("review-output");
const hypothesisOutputEl = document.getElementById("hypothesis-output");
const referencesOutputEl = document.getElementById("references-output");
const generationOutputEl = document.getElementById("generation-output");

const dbSizeEl = document.getElementById("db-size");
const chunksSizeEl = document.getElementById("chunks-size");
const documentsCountEl = document.getElementById("documents-count");
const chunksCountEl = document.getElementById("chunks-count");
const refsCountEl = document.getElementById("refs-count");

let statsTimer = null;
let generationTimer = null;
let activityTimer = null;
let currentGenerationRequestId = "";
let lastHeartbeatAtMs = 0;
let lastActivityStage = "review_prompt";
let lastActivityElapsedSec = 0;

const STAGE_LABELS = {
  review_prompt: "Reviewing prompt",
  similarity_search: "Running similarity search",
  context_build: "Building context blocks",
  llama_generation: "Generating with Llama 3",
  completed_or_unknown: "Finalizing response",
};

function parseLineSeparated(value) {
  return value
    .split("\n")
    .map((x) => x.trim())
    .filter(Boolean);
}

function setStatus(text) {
  statusEl.textContent = text;
}

function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  if (mins <= 0) {
    return `${secs}s`;
  }
  return `${mins}m ${secs}s`;
}

function stageLabel(stage) {
  return STAGE_LABELS[String(stage || "").trim()] || "Generating with Llama 3";
}

function setProgress(percent, label, isError = false) {
  const clamped = Math.max(0, Math.min(100, Number(percent) || 0));
  progressFillEl.style.width = `${clamped}%`;
  progressLabelEl.textContent = `${Math.round(clamped)}% - ${label}`;
  if (isError) {
    progressWrapEl.classList.add("error");
  } else {
    progressWrapEl.classList.remove("error");
  }
}

function stopGenerationActivityPolling() {
  if (activityTimer !== null) {
    clearInterval(activityTimer);
    activityTimer = null;
  }
  currentGenerationRequestId = "";
  lastHeartbeatAtMs = 0;
  lastActivityStage = "review_prompt";
  lastActivityElapsedSec = 0;
}

async function pollGenerationActivity(requestId) {
  try {
    const response = await fetch(`/api/hypothesis/activity?request_id=${encodeURIComponent(requestId)}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`activity ${response.status}`);
    }
    const data = await response.json();
    const activity = data.activity || {};
    if (String(activity.request_id || "") !== requestId) {
      return;
    }
    lastHeartbeatAtMs = Date.now();
    lastActivityStage = String(activity.stage || "llama_generation");
    const elapsed = Number(activity.elapsed_seconds);
    if (Number.isFinite(elapsed) && elapsed >= 0) {
      lastActivityElapsedSec = elapsed;
    }
  } catch (err) {
    // Keep previous heartbeat values so UI can indicate stale heartbeat.
  }
}

function startGenerationActivityPolling(requestId) {
  stopGenerationActivityPolling();
  currentGenerationRequestId = requestId;
  void pollGenerationActivity(requestId);
  activityTimer = setInterval(() => {
    if (!currentGenerationRequestId) {
      return;
    }
    void pollGenerationActivity(currentGenerationRequestId);
  }, 2000);
}

function stopGenerationProgress() {
  if (generationTimer !== null) {
    clearInterval(generationTimer);
    generationTimer = null;
  }
  stopGenerationActivityPolling();
}

function startGenerationProgress(requestId) {
  stopGenerationProgress();
  const startedAt = Date.now();
  startGenerationActivityPolling(requestId);
  setProgress(3, "Reviewing prompt");

  generationTimer = setInterval(() => {
    const elapsed = (Date.now() - startedAt) / 1000;
    let target = 3;
    let label = "Reviewing prompt";

    if (elapsed >= 1.2 && elapsed < 3.0) {
      target = 18 + (elapsed - 1.2) * 8;
      label = "Running similarity search";
    } else if (elapsed >= 3.0 && elapsed < 6.0) {
      target = 32 + (elapsed - 3.0) * 7;
      label = "Building context blocks";
    } else if (elapsed >= 6.0) {
      target = 54 + (elapsed - 6.0) * 4.5;
      const heartbeatAgeSec = lastHeartbeatAtMs ? (Date.now() - lastHeartbeatAtMs) / 1000 : Number.POSITIVE_INFINITY;
      const liveElapsed = lastActivityElapsedSec > 0 ? lastActivityElapsedSec : elapsed;
      if (heartbeatAgeSec <= 12) {
        label = `${stageLabel(lastActivityStage)} (${formatElapsed(liveElapsed)} elapsed)`;
        setStatus(`Generating... server active (${formatElapsed(liveElapsed)} elapsed).`);
      } else {
        label = `Generating with Llama 3 (${formatElapsed(elapsed)} elapsed, waiting for heartbeat)`;
        setStatus(`Generating... waiting for server heartbeat (${formatElapsed(elapsed)} elapsed).`);
      }
    }

    const current = parseFloat((progressFillEl.style.width || "0").replace("%", "")) || 0;
    let next = Math.min(94, Math.max(current, target));
    if (elapsed >= 12) {
      const pulse = 93 + (Math.sin(elapsed * 2) + 1) * 1.0; // visual activity near completion
      next = Math.max(next, pulse);
      next = Math.min(95, next);
    }
    setProgress(next, label);
  }, 250);
}

async function generateHypothesis(payload) {
  const response = await fetch("/api/hypothesis/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.message || `generate failed (${response.status})`);
  }
  return data;
}

function renderReview(review) {
  const terms = Array.isArray(review.key_terms) ? review.key_terms : [];
  const notes = Array.isArray(review.notes) ? review.notes : [];
  const warnings = Array.isArray(review.warnings) ? review.warnings : [];
  const lines = [
    `cleaned_prompt: ${review.cleaned_prompt || ""}`,
    "",
    `key_terms: ${terms.length ? terms.join(", ") : "(none)"}`,
    "",
    "notes:",
    ...(notes.length ? notes.map((n) => `- ${n}`) : ["- (none)"]),
    "",
    "warnings:",
    ...(warnings.length ? warnings.map((w) => `- ${w}`) : ["- (none)"]),
  ];
  reviewOutputEl.textContent = lines.join("\n");
}

function renderReferences(refs) {
  if (!Array.isArray(refs) || refs.length === 0) {
    referencesOutputEl.textContent = "No references retrieved.";
    return;
  }

  const lines = [];
  refs.forEach((ref) => {
    const distance = Number(ref.distance);
    const distanceLabel = Number.isFinite(distance) ? distance.toFixed(6) : "?";
    const weightedDistance = Number(ref.weighted_distance);
    const weightedLabel = Number.isFinite(weightedDistance) ? weightedDistance.toFixed(6) : "?";
    lines.push(`[${ref.id}] distance=${distanceLabel}`);
    lines.push(`weighted_distance: ${weightedLabel}`);
    lines.push(`citation_count: ${Number(ref.citation_count || 0)}`);
    lines.push(`title: ${ref.title || "(untitled)"}`);
    lines.push(`section: ${ref.section || "(none)"}`);
    lines.push(`url: ${ref.url || "(none)"}`);
    lines.push(`source_url: ${ref.source_url || "(none)"}`);
    lines.push(`doi: ${ref.doi || "(none)"}`);
    lines.push(`venue: ${ref.venue || "(none)"}`);
    lines.push(`published_date: ${ref.published_date || "(none)"}`);
    lines.push(`authors: ${ref.authors || "(none)"}`);
    lines.push(`excerpt: ${ref.excerpt || "(none)"}`);
    lines.push("");
  });
  referencesOutputEl.textContent = lines.join("\n");
}

async function refreshStats() {
  try {
    const response = await fetch("/api/hypothesis/stats", { cache: "no-store" });
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptEl.value.trim();
  if (!prompt) {
    setStatus("Enter a prompt first.");
    return;
  }

  const payload = {
    prompt,
    top_k: Number.isFinite(Number(topKEl.value)) ? Number(topKEl.value) : 8,
    recent_only: Boolean(recentOnlyEl.checked),
    document_urls: parseLineSeparated(urlFilterEl.value),
  };
  const requestId = `req_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  payload.request_id = requestId;

  runBtn.disabled = true;
  setStatus("Generating...");
  startGenerationProgress(requestId);
  try {
    const data = await generateHypothesis(payload);
    queryOutputEl.textContent = data.search_query || "-";
    renderReview(data.review || {});
    hypothesisOutputEl.textContent = data.hypothesis || "-";
    renderReferences(data.references || []);
    generationOutputEl.textContent = JSON.stringify(
      {
        generation_mode: data.generation_mode || "unknown",
        request_id: data.request_id || requestId,
        llm: data.llm || {},
      },
      null,
      2
    );
    stopGenerationProgress();
    setProgress(100, "Completed");
    setStatus(`Generated hypothesis with ${data.retrieved_count ?? 0} references.`);
  } catch (err) {
    stopGenerationProgress();
    setProgress(100, "Failed", true);
    setStatus(err.message || "Generation failed.");
  } finally {
    runBtn.disabled = false;
  }
});

setProgress(0, "Idle");
refreshStats();
scheduleStatsPolling();
