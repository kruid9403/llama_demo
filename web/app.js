const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const promptInput = document.getElementById("prompt");
const urlInput = document.getElementById("source-url");
const sendButton = document.getElementById("send");
const statusText = document.getElementById("status-text");

const history = [];

function addMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;
  bubble.textContent = text;
  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  promptInput.disabled = isBusy;
  sendButton.textContent = isBusy ? "Thinking..." : "Send";
}

function setStatus(text) {
  statusText.textContent = text;
}

async function ask(question, url) {
  setBusy(true);
  setStatus("Sending request...");
  const assistantBubble = addMessage("assistant", "");
  try {
    const response = await fetch("/api/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question, history, url }),
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let answer = "";
    let pending = "";

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

      const liveTail = pending.startsWith("[[") ? "" : pending;
      assistantBubble.textContent = (answer + liveTail).trimEnd();
      chat.scrollTop = chat.scrollHeight;
    }

    if (pending) {
      if (pending.startsWith("[[STATUS]] ")) {
        setStatus(pending.replace("[[STATUS]] ", ""));
      } else {
        answer += pending;
      }
    }
    assistantBubble.textContent = answer.trimEnd();

    if (!answer) {
      assistantBubble.textContent = "No response received.";
    }
    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: answer || "No response received." });
  } catch (err) {
    assistantBubble.textContent = "Something went wrong. Check the server logs.";
    setStatus("Request failed.");
  } finally {
    setBusy(false);
    if (statusText.textContent !== "Request failed.") {
      setStatus("Idle");
    }
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
