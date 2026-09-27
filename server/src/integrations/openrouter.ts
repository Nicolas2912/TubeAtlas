// The only module that talks to OpenRouter. Native fetch, no SDK, and deliberately no retries:
// a repeated paid call is the caller's (usually the user's) decision.
import { AppError } from '../errors.ts';

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

export type Usage = {
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number;
  /** USD as reported by OpenRouter; null when not reported. */
  cost: number | null;
};

export type ChatResult = {
  text: string;
  model: string;
  provider: string | null;
  finishReason: string | null;
  usage: Usage | null;
  generationId: string | null;
  latencyMs: number;
};

export type StructuredResult = {
  json: unknown;
  model: string;
  provider: string | null;
  usage: Usage | null;
  generationId: string | null;
  latencyMs: number;
};

export type EmbedResult = { vectors: Float32Array[]; model: string; usage: Usage | null };

export type ReasoningEffort = 'minimal' | 'low' | 'medium' | 'high';

export type OpenRouter = ReturnType<typeof createOpenRouter>;

const TIMEOUT_MS = { chat: 120_000, embed: 60_000, structured: 600_000 } as const;

export function createOpenRouter(options: {
  apiKey: string;
  baseUrl: string;
  fetch?: typeof globalThis.fetch;
  /** Overrides for tests. */
  timeoutsMs?: Partial<Record<keyof typeof TIMEOUT_MS, number>>;
}) {
  const { apiKey, baseUrl } = options;
  const fetchFn = options.fetch ?? globalThis.fetch;
  const timeouts = { ...TIMEOUT_MS, ...options.timeoutsMs };

  /**
   * POSTs JSON and hands the response to `read`, all under one timeout. Transport, timeout, and HTTP
   * failures become AppErrors; an abort by the caller propagates unchanged as an AbortError.
   */
  async function call<T>(path: string, body: unknown, timeoutMs: number, signal: AbortSignal | undefined, read: (res: Response) => Promise<T>): Promise<T> {
    const timeout = AbortSignal.timeout(timeoutMs);
    try {
      const res = await send(path, body, signal ? AbortSignal.any([signal, timeout]) : timeout);
      return await read(res);
    } catch (err) {
      if (err instanceof AppError) throw err;
      if (signal?.aborted) throw err;
      if (timeout.aborted) throw new AppError(504, 'PROVIDER_TIMEOUT', 'OpenRouter did not answer in time.', true);
      throw new AppError(502, 'PROVIDER_UNREACHABLE', 'Could not reach OpenRouter.', true);
    }
  }

  async function send(path: string, body: unknown, signal: AbortSignal): Promise<Response> {
    const res = await fetchFn(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json', 'X-Title': 'TubeAtlas' },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new AppError(
        502,
        `PROVIDER_HTTP_${res.status}`,
        `OpenRouter returned HTTP ${res.status}${providerMessage(detail)}`,
        res.status === 429 || res.status >= 500,
      );
    }
    return res;
  }

  async function chat(args: { model: string; messages: ChatMessage[]; maxTokens?: number; signal?: AbortSignal }): Promise<ChatResult> {
    const started = Date.now();
    const body = await call('/chat/completions', { model: args.model, messages: args.messages, max_tokens: args.maxTokens }, timeouts.chat, args.signal, readJson);
    const choice = firstChoice(body);
    return {
      text: typeof choice.message?.content === 'string' ? choice.message.content : '',
      model: String(body.model ?? args.model),
      provider: typeof body.provider === 'string' ? body.provider : null,
      finishReason: choice.finish_reason ?? null,
      usage: normalizeUsage(body.usage),
      generationId: typeof body.id === 'string' ? body.id : null,
      latencyMs: Date.now() - started,
    };
  }

  /** Streams a chat completion; onDelta receives text as it arrives. Resolves with the full result. */
  async function chatStream(args: {
    model: string;
    messages: ChatMessage[];
    maxTokens?: number;
    signal?: AbortSignal;
    onDelta: (text: string) => void;
  }): Promise<ChatResult> {
    const started = Date.now();
    const result: ChatResult = { text: '', model: args.model, provider: null, finishReason: null, usage: null, generationId: null, latencyMs: 0 };
    const body = { model: args.model, messages: args.messages, max_tokens: args.maxTokens, stream: true };
    await call('/chat/completions', body, timeouts.chat, args.signal, async (res) => {
      if (!res.body) throw new AppError(502, 'PROVIDER_BAD_RESPONSE', 'OpenRouter returned an empty stream.', true);
      for await (const data of sseData(res.body)) {
        if (data === '[DONE]') break;
        let chunk: any;
        try {
          chunk = JSON.parse(data);
        } catch {
          throw new AppError(502, 'PROVIDER_BAD_RESPONSE', 'OpenRouter sent an unreadable stream chunk.', true);
        }
        if (chunk.error) {
          throw new AppError(502, 'PROVIDER_ERROR', `OpenRouter reported an error mid-stream${providerMessage(JSON.stringify(chunk))}`, true);
        }
        if (typeof chunk.id === 'string') result.generationId = chunk.id;
        if (typeof chunk.model === 'string') result.model = chunk.model;
        if (typeof chunk.provider === 'string') result.provider = chunk.provider;
        if (chunk.usage) result.usage = normalizeUsage(chunk.usage);
        const choice = chunk.choices?.[0];
        if (choice?.finish_reason) result.finishReason = choice.finish_reason;
        const delta = choice?.delta?.content;
        if (typeof delta === 'string' && delta !== '') {
          result.text += delta;
          args.onDelta(delta);
        }
      }
    });
    result.latencyMs = Date.now() - started;
    return result;
  }

  /** Embeds texts; vectors come back in input order and L2-normalized, so cosine similarity is a dot product. */
  async function embed(args: { model: string; input: string[]; signal?: AbortSignal }): Promise<EmbedResult> {
    if (args.input.length === 0) return { vectors: [], model: args.model, usage: null };
    const body = await call('/embeddings', { model: args.model, input: args.input }, timeouts.embed, args.signal, readJson);
    const data = Array.isArray(body.data) ? [...body.data].sort((a: any, b: any) => a.index - b.index) : [];
    if (data.length !== args.input.length) {
      throw new AppError(502, 'PROVIDER_BAD_RESPONSE', `Expected ${args.input.length} embeddings, got ${data.length}.`, true);
    }
    const vectors = data.map((item: any) => normalize(item.embedding));
    const dim = vectors[0]!.length;
    if (vectors.some((v: Float32Array) => v.length !== dim)) {
      throw new AppError(502, 'PROVIDER_BAD_RESPONSE', 'Embeddings have inconsistent dimensions.', true);
    }
    return { vectors, model: String(body.model ?? args.model), usage: normalizeUsage(body.usage) };
  }

  /** Strict JSON-schema output with explicit reasoning effort; refuses anything but a complete answer from the requested model. */
  async function structured(args: {
    model: string;
    reasoningEffort: ReasoningEffort;
    schemaName: string;
    jsonSchema: object;
    messages: ChatMessage[];
    maxTokens: number;
    seed: number;
    signal?: AbortSignal;
  }): Promise<StructuredResult> {
    const started = Date.now();
    const body = await call(
      '/chat/completions',
      {
        model: args.model,
        reasoning: { effort: args.reasoningEffort, exclude: true },
        provider: { require_parameters: true },
        seed: args.seed,
        max_tokens: args.maxTokens,
        response_format: { type: 'json_schema', json_schema: { name: args.schemaName, strict: true, schema: args.jsonSchema } },
        messages: args.messages,
      },
      timeouts.structured,
      args.signal,
      readJson,
    );
    const choice = firstChoice(body);
    if (choice.finish_reason === 'length') {
      throw new AppError(502, 'OUTPUT_TRUNCATED', 'The model ran out of output tokens before finishing.', false);
    }
    if (choice.finish_reason === 'content_filter') {
      throw new AppError(502, 'CONTENT_FILTERED', 'The provider filtered the model output.', false);
    }
    if (choice.message?.refusal) throw new AppError(502, 'MODEL_REFUSED', 'The model refused the request.', false);
    if (body.model !== args.model) {
      throw new AppError(502, 'MODEL_MISMATCH', `Requested ${args.model} but OpenRouter answered with ${String(body.model)}.`, false);
    }
    let json: unknown;
    try {
      json = JSON.parse(String(choice.message?.content ?? ''));
    } catch {
      throw new AppError(502, 'INVALID_JSON', 'The model output is not valid JSON.', false);
    }
    return {
      json,
      model: body.model,
      provider: typeof body.provider === 'string' ? body.provider : null,
      usage: normalizeUsage(body.usage),
      generationId: typeof body.id === 'string' ? body.id : null,
      latencyMs: Date.now() - started,
    };
  }

  return { chat, chatStream, embed, structured };
}

/** Yields the data payload of each SSE event; ignores comment lines (OpenRouter keep-alives) and other fields. */
export async function* sseData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buffer = '';
  let data: string[] = [];
  for await (const bytes of body) {
    buffer += decoder.decode(bytes, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newline).replace(/\r$/, '');
      buffer = buffer.slice(newline + 1);
      if (line === '') {
        if (data.length) yield data.join('\n');
        data = [];
      } else if (line.startsWith('data:')) {
        data.push(line.slice(5).replace(/^ /, ''));
      }
      // Lines starting with ':' are comments; other fields (event:, id:) are unused.
    }
  }
  buffer += decoder.decode();
  const last = buffer.replace(/\r$/, '');
  if (last.startsWith('data:')) data.push(last.slice(5).replace(/^ /, ''));
  if (data.length) yield data.join('\n');
}

async function readJson(res: Response): Promise<any> {
  const text = await res.text(); // transport errors and aborts propagate to call()
  try {
    return JSON.parse(text);
  } catch {
    throw new AppError(502, 'PROVIDER_BAD_RESPONSE', 'OpenRouter returned a response that is not JSON.', true);
  }
}

function firstChoice(body: any) {
  const choice = body?.choices?.[0];
  if (!choice) {
    // OpenRouter can return 200 with an error object instead of choices.
    throw new AppError(502, 'PROVIDER_ERROR', `OpenRouter returned no answer${providerMessage(JSON.stringify(body ?? {}))}`, true);
  }
  return choice;
}

/** A short, single-line excerpt of the provider's own error message (never our request, never the key). */
function providerMessage(raw: string): string {
  let message = '';
  try {
    const parsed = JSON.parse(raw);
    message = String(parsed?.error?.message ?? '');
  } catch {
    message = '';
  }
  message = message.replace(/\s+/g, ' ').trim().slice(0, 300);
  return message ? `: ${message}` : '.';
}

function normalizeUsage(raw: any): Usage | null {
  if (!raw || typeof raw !== 'object') return null;
  return {
    promptTokens: Number(raw.prompt_tokens ?? 0),
    completionTokens: Number(raw.completion_tokens ?? 0),
    reasoningTokens: Number(raw.completion_tokens_details?.reasoning_tokens ?? 0),
    cost: typeof raw.cost === 'number' ? raw.cost : null,
  };
}

function normalize(values: unknown): Float32Array {
  if (!Array.isArray(values) || values.length === 0 || !values.every((v) => typeof v === 'number' && Number.isFinite(v))) {
    throw new AppError(502, 'PROVIDER_BAD_RESPONSE', 'OpenRouter returned an invalid embedding.', true);
  }
  const vector = Float32Array.from(values as number[]);
  let sumOfSquares = 0;
  for (const v of vector) sumOfSquares += v * v;
  const norm = Math.sqrt(sumOfSquares);
  if (norm === 0) throw new AppError(502, 'PROVIDER_BAD_RESPONSE', 'OpenRouter returned a zero embedding.', true);
  for (let i = 0; i < vector.length; i++) vector[i] /= norm;
  return vector;
}
