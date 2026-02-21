const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const promptInput = document.getElementById("prompt");
const urlInput = document.getElementById("source-url");
const sendButton = document.getElementById("send");
const interruptButton = document.getElementById("interrupt");
const statusText = document.getElementById("status-text");
const vectorSizeText = document.getElementById("vector-size");

const history = [];
let autoScrollEnabled = true;
let activeController = null;
let streamActive = false;

function isNearBottom(el, threshold = 80) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function maybeScrollToBottom(force = false) {
  if (force || autoScrollEnabled) {
    chat.scrollTop = chat.scrollHeight;
  }
}

function addMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;
  bubble.textContent = text;
  chat.appendChild(bubble);
  maybeScrollToBottom(true);
  return bubble;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  promptInput.disabled = isBusy;
  urlInput.disabled = isBusy;
  interruptButton.disabled = !isBusy;
  sendButton.textContent = isBusy ? "Thinking..." : "Send";
}

function setStatus(text) {
  statusText.textContent = text;
}

async function refreshVectorStats() {
  if (!vectorSizeText) {
    return;
  }
  try {
    const response = await fetch("/api/stats", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`stats ${response.status}`);
    }
    const data = await response.json();
    const stats = data?.stats;
    if (!stats) {
      throw new Error("missing stats");
    }
    vectorSizeText.textContent = `Vector DB: ${stats.chunks} chunks (${stats.chunk_table_size})`;
  } catch (err) {
    vectorSizeText.textContent = "Vector DB: unavailable";
  }
}

async function ask(question, url) {
  const controller = new AbortController();
  activeController = controller;
  streamActive = true;
  setBusy(true);
  setStatus("Sending request...");
  const assistantBubble = addMessage("assistant", "");
  try {
    const response = await fetch("/api/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify({ question, history, url }),
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let answer = "";
    let pending = "";
    let rafScheduled = false;
    let latestLiveTail = "";

    const render = () => {
      rafScheduled = false;
      assistantBubble.textContent = (answer + latestLiveTail).trimEnd();
      maybeScrollToBottom();
    };

    const scheduleRender = () => {
      if (rafScheduled) {
        return;
      }
      rafScheduled = true;
      requestAnimationFrame(render);
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      const chunk = decoder.decode(value, { stream: true });
      pending += chunk;
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("[[STATUS]] ")) {
          setStatus(line.replace("[[STATUS]] ", ""));
          continue;
        }
        answer += line + "\n";
      }

      latestLiveTail = pending.startsWith("[[") ? "" : pending;
      scheduleRender();
    }

    if (pending) {
      if (pending.startsWith("[[STATUS]] ")) {
        setStatus(pending.replace("[[STATUS]] ", ""));
      } else {
        answer += pending;
      }
    }
    latestLiveTail = "";
    render();

    if (!answer) {
      assistantBubble.textContent = "No response received.";
    }
    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: answer || "No response received." });
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus("Interrupted.");
      if (!assistantBubble.textContent.trim()) {
        assistantBubble.textContent = "Interrupted.";
      }
    } else {
      assistantBubble.textContent = "Something went wrong. Check the server logs.";
      setStatus("Request failed.");
    }
  } finally {
    activeController = null;
    streamActive = false;
    setBusy(false);
    refreshVectorStats();
    if (statusText.textContent !== "Request failed.") {
      setStatus("Idle");
    }
  }
}

async function interruptRun() {
  setStatus("Interrupt requested...");
  if (streamActive && activeController) {
    activeController.abort();
  }
  try {
    const response = await fetch("/api/interrupt", {
      method: "POST",
      keepalive: true,
    });
    if (!response.ok) {
      setStatus(`Interrupt failed (${response.status}).`);
      return;
    }
    setStatus("Interrupt signal sent.");
  } catch (err) {
    setStatus("Interrupt request failed.");
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = promptInput.value.trim();
  if (!question) {
    return;
  }
  const url = urlInput.value.trim();
  addMessage("user", question);
  promptInput.value = "";
  ask(question, url);
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

interruptButton.addEventListener("click", () => {
  interruptRun();
});

chat.addEventListener("scroll", () => {
  autoScrollEnabled = isNearBottom(chat);
});

refreshVectorStats();
setInterval(refreshVectorStats, 15000);
