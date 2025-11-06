export type SSEEvent = {
  event: string;
  [key: string]: unknown;
};

function parseEventBlock(block: string): SSEEvent | null {
  const lines = block.split("\n");
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  const payload = dataLines.join("\n");
  if (payload === "[DONE]") {
    return { event: "done" };
  }
  try {
    const parsed = JSON.parse(payload);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as SSEEvent;
    }
  } catch (err) {
    console.warn("Failed to parse SSE payload", err, payload);
  }
  return null;
}

export async function streamSSE(
  url: string,
  body: unknown,
  onEvent: (event: SSEEvent) => void,
  options?: { signal?: AbortSignal }
): Promise<void> {
  const localController = options?.signal ? null : new AbortController();
  const signal = options?.signal ?? localController?.signal;
  const timeoutId = localController
    ? setTimeout(() => localController.abort(), 900000)
    : undefined;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    if (localController) {
      localController.abort();
    }
    throw new Error(`Request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const flushBuffer = () => {
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseEventBlock(block.trim());
      if (event) {
        onEvent(event);
      }
      boundary = buffer.indexOf("\n\n");
    }
  };

  const heartbeat = setInterval(() => {
    console.debug("🔄 SSE connection alive");
  }, 30000);

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      flushBuffer();
    }

    buffer += decoder.decode();
    flushBuffer();
  } finally {
    clearInterval(heartbeat);
    if (typeof timeoutId !== "undefined") {
      clearTimeout(timeoutId);
    }
  }
}

