const form = document.getElementById("start-form");
const seedUrlsEl = document.getElementById("seed-urls");
const domainsEl = document.getElementById("domains");
const rawDirEl = document.getElementById("raw-dir");
const maxPagesEl = document.getElementById("max-pages");
const maxDepthEl = document.getElementById("max-depth");
const delayEl = document.getElementById("delay-seconds");
const respectRobotsEl = document.getElementById("respect-robots");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const requestStatus = document.getElementById("request-status");

const runStateEl = document.getElementById("run-state");
const pagesEl = document.getElementById("pages-ingested");
const chunksEl = document.getElementById("chunks-inserted");
const errSkipEl = document.getElementById("err-skip");
const runtimeEl = document.getElementById("runtime");
const dbStatsEl = document.getElementById("db-stats");
const logView = document.getElementById("log-view");
const clearLogBtn = document.getElementById("clear-log");

let pollingTimer = null;
let latestLogSig = "";

function setRequestStatus(text) {
  requestStatus.textContent = text;
}

function formatRuntime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 1) return "0s";
  const s = Math.floor(seconds);
  const mins = Math.floor(s / 60);
  const rem = s % 60;
  return mins > 0 ? `${mins}m ${rem}s` : `${rem}s`;
}

function parseLineSeparated(value) {
  return value
    .split("\n")
    .map((x) => x.trim())
    .filter(Boolean);
}

function parseCommaSeparated(value) {
  return value
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

async function startSpider(payload) {
  const response = await fetch("/api/spider/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.message || `start failed (${response.status})`);
  }
  return data;
}

async function stopSpider() {
  const response = await fetch("/api/spider/stop", { method: "POST" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.ok) {
    throw new Error(data.message || `stop failed (${response.status})`);
  }
  return data;
}

function renderLogs(logs) {
  const lines = logs.map((entry) => {
    const ts = entry.ts ? new Date(entry.ts).toLocaleTimeString() : "--:--:--";
    return `${ts} ${entry.message}`;
  });
  const text = lines.join("\n");
  const sig = `${logs.length}:${logs.length ? logs[logs.length - 1].ts : ""}`;
  if (sig === latestLogSig) {
    return;
  }
  latestLogSig = sig;
  const nearBottom = logView.scrollHeight - logView.scrollTop - logView.clientHeight < 80;
  logView.textContent = text;
  if (nearBottom) {
    logView.scrollTop = logView.scrollHeight;
  }
}

function renderState(state) {
  runStateEl.textContent = state.status || "Idle";
  pagesEl.textContent = String(state.pages_ingested ?? 0);
  chunksEl.textContent = String(state.chunks_inserted ?? 0);
  errSkipEl.textContent = `${state.errors ?? 0} / ${state.skips ?? 0}`;
  runtimeEl.textContent = formatRuntime(state.runtime_seconds ?? 0);

  if (state.db_stats) {
    const size = state.db_stats.database_size || "?";
    const chunkSize = state.db_stats.chunks_total_size || "?";
    dbStatsEl.textContent = `${size} (chunks ${chunkSize}) | ${state.db_stats.documents} / ${state.db_stats.chunks} / ${state.db_stats.references}`;
  } else if (state.db_error) {
    dbStatsEl.textContent = "unavailable";
  }

  const running = Boolean(state.running);
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  if (running) {
    setRequestStatus("Running...");
  }

  if (Array.isArray(state.logs)) {
    renderLogs(state.logs);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/spider/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const data = await response.json();
    renderState(data.state || {});
  } catch (err) {
    setRequestStatus("Status unavailable");
  }
}

function schedulePolling() {
  if (pollingTimer !== null) {
    clearInterval(pollingTimer);
  }
  pollingTimer = setInterval(refreshStatus, 10000);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const seedUrls = parseLineSeparated(seedUrlsEl.value);

  const payload = {
    seed_urls: seedUrls,
    allowed_domains: parseCommaSeparated(domainsEl.value),
    max_pages: Number.isFinite(Number(maxPagesEl.value)) ? Number(maxPagesEl.value) : 0,
    max_depth: Number.isFinite(Number(maxDepthEl.value)) ? Number(maxDepthEl.value) : -1,
    delay_seconds: Number.isFinite(Number(delayEl.value)) ? Number(delayEl.value) : 1.0,
    respect_robots: respectRobotsEl.checked,
    raw_dir: rawDirEl.value.trim(),
  };

  setRequestStatus("Starting spider...");
  try {
    await startSpider(payload);
    setRequestStatus("Spider started.");
    await refreshStatus();
  } catch (err) {
    setRequestStatus(err.message || "Start failed.");
  }
});

stopBtn.addEventListener("click", async () => {
  setRequestStatus("Stopping spider...");
  try {
    await stopSpider();
    setRequestStatus("Stop requested.");
    await refreshStatus();
  } catch (err) {
    setRequestStatus(err.message || "Stop failed.");
  }
});

clearLogBtn.addEventListener("click", () => {
  logView.textContent = "";
  latestLogSig = "";
});

refreshStatus();
schedulePolling();
