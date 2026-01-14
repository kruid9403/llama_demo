const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const promptInput = document.getElementById("prompt");
const urlInput = document.getElementById("source-url");
const sendButton = document.getElementById("send");

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

async function ask(question, url) {
  setBusy(true);
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

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      const chunk = decoder.decode(value, { stream: true });
      answer += chunk;
      assistantBubble.textContent = answer;
      chat.scrollTop = chat.scrollHeight;
    }

    if (!answer) {
      assistantBubble.textContent = "No response received.";
    }
    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: answer || "No response received." });
  } catch (err) {
    assistantBubble.textContent = "Something went wrong. Check the server logs.";
  } finally {
    setBusy(false);
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
