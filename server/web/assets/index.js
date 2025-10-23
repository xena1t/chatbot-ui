const state = {
  messages: [],
  isStreaming: false,
  error: null,
  previewUrl: null,
};

let abortController = null;
let currentRequest = null;

const chatPane = document.getElementById("chat-pane");
const errorBox = document.getElementById("error-box");
const form = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt");
const videoInput = document.getElementById("video-input");
const previewContainer = document.getElementById("preview-container");
const previewVideo = document.getElementById("video-preview");
const removeVideoButton = document.getElementById("remove-video");
const sendButton = document.getElementById("send-button");
const stopButton = document.getElementById("stop-button");
const fileLabelText = document.getElementById("file-label-text");

function makeId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

function renderMessages() {
  if (!chatPane) return;
  chatPane.innerHTML = "";
  if (state.messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Start by sending a message or video.";
    chatPane.appendChild(empty);
    return;
  }

  for (const message of state.messages) {
    const article = document.createElement("article");
    article.className = `message ${message.role}`;

    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "You" : "Assistant";
    article.appendChild(role);

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const text = document.createElement("p");
    if (message.content && message.content.trim().length > 0) {
      text.textContent = message.content;
    } else {
      text.className = "muted";
      text.textContent = state.isStreaming && message.role === "assistant" ? "Waiting for response…" : "";
    }
    bubble.appendChild(text);

    if (message.videoUrl) {
      const wrapper = document.createElement("div");
      wrapper.className = "attachment";
      const video = document.createElement("video");
      video.src = message.videoUrl;
      video.controls = true;
      video.preload = "metadata";
      wrapper.appendChild(video);
      bubble.appendChild(wrapper);
    }

    if (message.frames && message.frames.length > 0) {
      const frames = document.createElement("div");
      frames.className = "frames";
      for (const frame of message.frames) {
        const img = document.createElement("img");
        img.src = frame;
        img.alt = "Video frame";
        frames.appendChild(img);
      }
      bubble.appendChild(frames);
    }

    article.appendChild(bubble);
    chatPane.appendChild(article);
  }

  chatPane.scrollTop = chatPane.scrollHeight;
}

function setError(message) {
  state.error = message;
  if (!errorBox) return;
  if (message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  } else {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }
}

function updateSendButtonState() {
  const hasText = promptInput && promptInput.value.trim().length > 0;
  const hasVideo = videoInput && videoInput.files && videoInput.files.length > 0;
  if (sendButton) {
    sendButton.disabled = state.isStreaming || (!hasText && !hasVideo);
  }
}

function setStreaming(flag) {
  state.isStreaming = flag;
  if (promptInput) promptInput.disabled = flag;
  if (videoInput) videoInput.disabled = flag;
  if (removeVideoButton) removeVideoButton.disabled = flag;
  if (stopButton) {
    if (flag) {
      stopButton.classList.remove("hidden");
      stopButton.disabled = false;
    } else {
      stopButton.classList.add("hidden");
      stopButton.disabled = true;
    }
  }
  updateSendButtonState();
  renderMessages();
}

function resetVideoSelection() {
  if (videoInput) {
    videoInput.value = "";
  }
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
  if (previewVideo) {
    previewVideo.removeAttribute("src");
    previewVideo.load?.();
  }
  if (previewContainer) {
    previewContainer.classList.add("hidden");
  }
  if (fileLabelText) {
    fileLabelText.textContent = "Attach MP4/MOV (≤30s)";
  }
  updateSendButtonState();
}

function handleVideoChange() {
  const file = videoInput && videoInput.files ? videoInput.files[0] : null;
  if (!file) {
    resetVideoSelection();
    return;
  }
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }
  const url = URL.createObjectURL(file);
  state.previewUrl = url;
  if (previewVideo) {
    previewVideo.src = url;
    previewVideo.load?.();
  }
  if (previewContainer) {
    previewContainer.classList.remove("hidden");
  }
  if (fileLabelText) {
    fileLabelText.textContent = file.name;
  }
  updateSendButtonState();
}

function attachFramesToUser(userId, frames) {
  const message = state.messages.find((m) => m.id === userId);
  if (message) {
    message.frames = frames;
    renderMessages();
  }
}

function appendAssistantText(assistantId, text) {
  const message = state.messages.find((m) => m.id === assistantId);
  if (message) {
    message.content = (message.content || "") + text;
    renderMessages();
  }
}

function setAssistantContent(assistantId, content) {
  const message = state.messages.find((m) => m.id === assistantId);
  if (message) {
    message.content = content;
    renderMessages();
  }
}

async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  const text = await response.text();
  if (!response.ok) {
    let message = `Upload failed (${response.status})`;
    try {
      const parsed = JSON.parse(text);
      if (parsed && parsed.detail) {
        message = parsed.detail;
      }
    } catch (err) {
      if (text) message = text;
    }
    throw new Error(message);
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new Error("Failed to parse upload response");
  }
}

function parseEventBlock(block) {
  if (!block) return null;
  const lines = block.split("\n");
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n");
  if (payload === "[DONE]") {
    return { event: "done" };
  }
  try {
    const parsed = JSON.parse(payload);
    if (parsed && typeof parsed === "object") {
      return parsed;
    }
  } catch (err) {
    console.warn("Failed to parse SSE payload", payload);
  }
  return null;
}

async function streamChat(body, onEvent, signal) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`Stream failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseEventBlock(chunk.trim());
      if (event) {
        onEvent(event);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  if (state.isStreaming) return;

  const text = promptInput ? promptInput.value.trim() : "";
  const file = videoInput && videoInput.files ? videoInput.files[0] : null;
  if (!text && !file) {
    setError("Enter a message or attach a short video.");
    return;
  }

  setError(null);

  let uploadResult = null;
  if (file) {
    try {
      uploadResult = await uploadVideo(file);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return;
    }
  }

  const userId = makeId();
  const assistantId = makeId();
  const conversation = state.messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
  conversation.push({ role: "user", content: text });

  state.messages.push({
    id: userId,
    role: "user",
    content: text,
    videoUrl: uploadResult ? uploadResult.video_url : null,
  });
  state.messages.push({ id: assistantId, role: "assistant", content: "" });
  renderMessages();

  currentRequest = { userId, assistantId };
  setStreaming(true);

  if (promptInput) promptInput.value = "";
  resetVideoSelection();

  abortController = new AbortController();

  try {
    await streamChat(
      {
        messages: conversation,
        video_url: uploadResult ? uploadResult.video_url : undefined,
      },
      (event) => {
        if (!currentRequest) return;
        if (event.event === "frames" && Array.isArray(event.frames)) {
          const frames = event.frames.filter((item) => typeof item === "string");
          attachFramesToUser(currentRequest.userId, frames);
        } else if (event.event === "token" && typeof event.text === "string") {
          appendAssistantText(currentRequest.assistantId, event.text);
        } else if (event.event === "error" && typeof event.message === "string") {
          setError(event.message);
          setAssistantContent(currentRequest.assistantId, `Error: ${event.message}`);
        }
      },
      abortController.signal
    );
  } catch (err) {
    if (currentRequest) {
      if (err && err.name === "AbortError") {
        setAssistantContent(currentRequest.assistantId, "Response cancelled.");
      } else {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        setAssistantContent(currentRequest.assistantId, `Error: ${message}`);
      }
    }
  } finally {
    currentRequest = null;
    abortController = null;
    setStreaming(false);
  }
}

function handleStop() {
  if (abortController) {
    abortController.abort();
  }
}

if (form) {
  form.addEventListener("submit", handleSubmit);
}
if (promptInput) {
  promptInput.addEventListener("input", () => {
    updateSendButtonState();
    setError(null);
  });
}
if (videoInput) {
  videoInput.addEventListener("change", () => {
    handleVideoChange();
    setError(null);
  });
}
if (removeVideoButton) {
  removeVideoButton.addEventListener("click", () => {
    resetVideoSelection();
  });
}
if (stopButton) {
  stopButton.addEventListener("click", handleStop);
}

updateSendButtonState();
renderMessages();
