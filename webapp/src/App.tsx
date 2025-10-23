import React, { FormEvent, useEffect, useRef, useState } from "react";
import { streamSSE, SSEEvent } from "./lib/sse";

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
  frames?: string[];
  videoUrl?: string;
};

type UploadResponse = {
  video_url: string;
  duration_s: number;
};

const makeId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });

  const payload = await response.text();
  if (!response.ok) {
    let message = `Upload failed (${response.status})`;
    try {
      const parsed = JSON.parse(payload);
      if (parsed?.detail) {
        message = parsed.detail;
      }
    } catch (err) {
      if (payload) {
        message = payload;
      }
    }
    throw new Error(message);
  }

  try {
    return JSON.parse(payload) as UploadResponse;
  } catch (err) {
    throw new Error("Failed to parse upload response");
  }
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const pendingIds = useRef<{ userId: string; assistantId: string } | null>(null);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (videoPreview) {
        URL.revokeObjectURL(videoPreview);
      }
    };
  }, [videoPreview]);

  const resetVideoSelection = () => {
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
    setVideoPreview(null);
    setVideoFile(null);
  };

  const handleVideoChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
    setVideoFile(file);
    setVideoPreview(file ? URL.createObjectURL(file) : null);
  };

  const updateAssistantMessage = (assistantId: string, text: string) => {
    setMessages((prev) => {
      const updated = prev.map((message) => {
        if (message.id === assistantId) {
          return { ...message, content: message.content + text };
        }
        return message;
      });
      return updated;
    });
  };

  const setAssistantError = (assistantId: string, message: string) => {
    setMessages((prev) => {
      const updated = prev.map((item) =>
        item.id === assistantId ? { ...item, content: message } : item
      );
      return updated;
    });
  };

  const attachFramesToUser = (userId: string, frames: string[]) => {
    setMessages((prev) =>
      prev.map((message) =>
        message.id === userId ? { ...message, frames } : message
      )
    );
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (isStreaming) {
      return;
    }

    const trimmed = input.trim();
    if (!trimmed && !videoFile) {
      setError("Enter a message or attach a short video.");
      return;
    }

    setError(null);

    let upload: UploadResponse | null = null;
    if (videoFile) {
      try {
        upload = await uploadVideo(videoFile);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        return;
      }
    }

    const userId = makeId();
    const assistantId = makeId();

    const userMessage: Message = {
      id: userId,
      role: "user",
      content: trimmed,
      videoUrl: upload?.video_url,
    };
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    resetVideoSelection();

    pendingIds.current = { userId, assistantId };
    setIsStreaming(true);

    const payloadMessages = [...messages, userMessage].map((message) => ({
      role: message.role,
      content: message.content,
    }));

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamSSE(
        "/api/chat/stream",
        {
          messages: payloadMessages,
          video_url: upload?.video_url,
        },
        (event: SSEEvent) => {
          const ids = pendingIds.current;
          if (!ids) {
            return;
          }
          if (event.event === "frames" && Array.isArray(event.frames)) {
            const frames = event.frames.filter((item) => typeof item === "string");
            attachFramesToUser(ids.userId, frames);
          } else if (event.event === "token" && typeof event.text === "string") {
            updateAssistantMessage(ids.assistantId, event.text);
          } else if (event.event === "error" && typeof event.message === "string") {
            setError(event.message);
            setAssistantError(ids.assistantId, `Error: ${event.message}`);
          }
        },
        { signal: controller.signal }
      );
    } catch (err) {
      const ids = pendingIds.current;
      if (ids) {
        if ((err as Error).name === "AbortError") {
          setAssistantError(ids.assistantId, "Response cancelled.");
        } else {
          const message = err instanceof Error ? err.message : String(err);
          setError(message);
          setAssistantError(ids.assistantId, `Error: ${message}`);
        }
      } else if (err instanceof Error) {
        setError(err.message);
      }
    } finally {
      pendingIds.current = null;
      abortRef.current = null;
      setIsStreaming(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>RunPod Qwen3-VL Chat</h1>
        <p>Upload a short video and chat with Qwen3-VL-32B via vLLM.</p>
      </header>

      <main className="layout">
        <section className="chat-pane">
          {messages.length === 0 ? (
            <div className="empty">Start by sending a message or video.</div>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="message-role">{message.role === "user" ? "You" : "Assistant"}</div>
                <div className="bubble">
                  {message.content ? (
                    <p>{message.content}</p>
                  ) : (
                    <p className="muted">Waiting for response…</p>
                  )}
                  {message.videoUrl && (
                    <div className="attachment">
                      <video src={message.videoUrl} controls preload="metadata" />
                    </div>
                  )}
                  {message.frames && message.frames.length > 0 && (
                    <div className="frames">
                      {message.frames.map((frame) => (
                        <img key={frame} src={frame} alt="Video frame" />
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))
          )}
        </section>

        <section className="composer">
          {error && <div className="error">{error}</div>}
          <form onSubmit={handleSubmit}>
            <label htmlFor="prompt">Message</label>
            <textarea
              id="prompt"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Describe what you want the model to do…"
              disabled={isStreaming}
              rows={4}
            />
            <div className="controls">
              <div className="file-input">
                <label className="file-label">
                  <span>{videoFile ? videoFile.name : "Attach MP4/MOV (≤30s)"}</span>
                  <input
                    type="file"
                    accept="video/mp4,video/quicktime"
                    onChange={handleVideoChange}
                    disabled={isStreaming}
                  />
                </label>
                {videoPreview && (
                  <div className="preview">
                    <video src={videoPreview} controls preload="metadata" />
                    <button type="button" onClick={resetVideoSelection} disabled={isStreaming}>
                      Remove
                    </button>
                  </div>
                )}
              </div>
              <div className="actions">
                {isStreaming && (
                  <button type="button" className="secondary" onClick={handleStop}>
                    Stop
                  </button>
                )}
                <button type="submit" disabled={isStreaming || (!input.trim() && !videoFile)}>
                  Send
                </button>
              </div>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
};

export default App;
