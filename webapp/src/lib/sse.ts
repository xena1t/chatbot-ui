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
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 900000); // 15 minutes

  if (options?.signal) {
    const abortSignal = options.signal;
    if (abortSignal.aborted) {
      controller.abort();
    } else {
      abortSignal.addEventListener("abort", () => controller.abort(), { once: true });
    }
  }

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: controller.signal,
  });

  if (!response.ok || !response.body) {
    controller.abort();
    throw new Error(`Request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  // Heartbeat to keep the connection open
  const heartbeat = setInterval(() => {
    console.debug("🔄 SSE connection alive");
  }, 30000);

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseEventBlock(block.trim());
        if (event) onEvent(event);
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    clearTimeout(timeout);
    clearInterval(heartbeat);
    controller.abort();
  }
}

