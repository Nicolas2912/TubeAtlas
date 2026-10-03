/** Fetch-based SSE framing, preserving split UTF-8, CRLF, multiline data, and a final event. */
export async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string; data: string }> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', event = 'message';
  let data: string[] = [];
  function line(value: string) {
    if (value.startsWith('data:')) data.push(value.slice(5).replace(/^ /u, ''));
    else if (value.startsWith('event:')) event = value.slice(6).trim() || 'message';
  }
  try {
    while (true) {
      const next = await reader.read();
      buffer += decoder.decode(next.value, { stream: !next.done });
      let newline: number;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const value = buffer.slice(0, newline).replace(/\r$/u, '');
        buffer = buffer.slice(newline + 1);
        if (!value) {
          if (data.length) yield { event, data: data.join('\n') };
          event = 'message'; data = [];
        } else line(value);
      }
      if (next.done) break;
    }
    if (buffer) line(buffer.replace(/\r$/u, ''));
    if (data.length) yield { event, data: data.join('\n') };
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
