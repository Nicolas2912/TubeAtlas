import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AppError } from '../errors.ts';
import { createOpenRouter, sseData } from './openrouter.ts';

const API_KEY = 'sk-or-test-SECRET-KEY';
const BASE = 'https://openrouter.test/api/v1';

type Call = { url: string; body: any; headers: Record<string, string> };

/** A fake fetch that answers from a queue, records every call, and honours abort signals. */
function fakeFetch(...responders: ((signal: AbortSignal) => Response | Promise<Response>)[]) {
  const calls: Call[] = [];
  const fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), body: JSON.parse(String(init?.body)), headers: Object.fromEntries(new Headers(init?.headers)) });
    const responder = responders[calls.length - 1];
    if (!responder) throw new Error('unexpected extra call (retry?)');
    const signal = init!.signal!;
    if (signal.aborted) throw signal.reason;
    return responder(signal);
  }) as typeof globalThis.fetch;
  return { fetch, calls };
}

const json = (body: unknown, status = 200) => () => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

/** A streaming response delivering the given pieces as separate reads; stays open if `hang` is set. */
function stream(pieces: string[], hang = false) {
  return (signal: AbortSignal) => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const piece of pieces) controller.enqueue(encoder.encode(piece));
        if (!hang) controller.close();
        signal.addEventListener('abort', () => controller.error(signal.reason));
      },
    });
    return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };
}

const client = (fetch: typeof globalThis.fetch, timeoutsMs?: Record<string, number>) => createOpenRouter({ apiKey: API_KEY, baseUrl: BASE, fetch, timeoutsMs });

async function rejectsWithCode(promise: Promise<unknown>, code: string, retryable?: boolean) {
  await assert.rejects(promise, (err: unknown) => {
    assert.ok(err instanceof AppError, `expected AppError, got ${String(err)}`);
    assert.equal(err.code, code);
    if (retryable !== undefined) assert.equal(err.retryable, retryable);
    assert.ok(!err.message.includes(API_KEY), 'error message must not contain the API key');
    return true;
  });
}

const collect = async (body: ReadableStream<Uint8Array>) => {
  const out: string[] = [];
  for await (const data of sseData(body)) out.push(data);
  return out;
};

test('sseData handles comments, pieces split mid-line and mid-CRLF, multi-line events, and a final unterminated event', async () => {
  const res = stream([
    ': OPENROUTER PROCESSING\n\n',
    'data: {"a":',
    '1}\r',
    '\n\r\ndata: first line\ndata: second line\n\nevent: ignored\nid: 7\ndata: x\n',
    '\n: keep-alive\n\ndata: tail',
  ])(new AbortController().signal);
  assert.deepEqual(await collect(res.body!), ['{"a":1}', 'first line\nsecond line', 'x', 'tail']);
});

test('sseData decodes multi-byte characters split across reads', async () => {
  const bytes = new TextEncoder().encode('data: "Übergrößen"\n\n');
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(bytes.slice(0, 9)); // splits the two-byte "Ü"
      c.enqueue(bytes.slice(9));
      c.close();
    },
  });
  assert.deepEqual(await collect(body), ['"Übergrößen"']);
});

test('chatStream delivers deltas in order and keeps model, provider, usage, and finish reason', async () => {
  const chunk = (obj: object) => `data: ${JSON.stringify(obj)}\n\n`;
  const { fetch, calls } = fakeFetch(
    stream([
      ': OPENROUTER PROCESSING\n\n',
      chunk({ id: 'gen-1', model: 'openai/gpt-4.1-mini', provider: 'Azure', choices: [{ delta: { role: 'assistant', content: '' } }] }),
      chunk({ id: 'gen-1', choices: [{ delta: { content: 'Hel' } }] }).slice(0, 20),
      chunk({ id: 'gen-1', choices: [{ delta: { content: 'Hel' } }] }).slice(20),
      chunk({ id: 'gen-1', choices: [{ delta: { content: 'lo' }, finish_reason: 'stop' }] }),
      chunk({ id: 'gen-1', choices: [], usage: { prompt_tokens: 9, completion_tokens: 2, cost: 0.00001 } }),
      'data: [DONE]\n\n',
      chunk({ choices: [{ delta: { content: 'ignored after DONE' } }] }),
    ]),
  );
  const deltas: string[] = [];
  const result = await client(fetch).chatStream({ model: 'openai/gpt-4.1-mini', messages: [{ role: 'user', content: 'Hi' }], onDelta: (d) => deltas.push(d) });

  assert.deepEqual(deltas, ['Hel', 'lo']);
  assert.equal(result.text, 'Hello');
  assert.equal(result.model, 'openai/gpt-4.1-mini');
  assert.equal(result.provider, 'Azure');
  assert.equal(result.finishReason, 'stop');
  assert.equal(result.generationId, 'gen-1');
  assert.deepEqual(result.usage, { promptTokens: 9, completionTokens: 2, reasoningTokens: 0, cost: 0.00001 });
  assert.equal(calls.length, 1);
  assert.equal(calls[0]!.url, `${BASE}/chat/completions`);
  assert.equal(calls[0]!.body.stream, true);
  assert.equal(calls[0]!.headers.authorization, `Bearer ${API_KEY}`);
  assert.equal(calls[0]!.headers['x-title'], 'TubeAtlas');
});

test('chatStream: a mid-stream error fails with PROVIDER_ERROR after the partial text was delivered', async () => {
  const { fetch, calls } = fakeFetch(
    stream([
      `data: ${JSON.stringify({ choices: [{ delta: { content: 'Partial' } }] })}\n\n`,
      `data: ${JSON.stringify({ error: { message: 'Upstream overloaded' }, choices: [{ finish_reason: 'error' }] })}\n\n`,
    ]),
  );
  const deltas: string[] = [];
  await rejectsWithCode(client(fetch).chatStream({ model: 'm', messages: [], onDelta: (d) => deltas.push(d) }), 'PROVIDER_ERROR', true);
  assert.deepEqual(deltas, ['Partial']);
  assert.equal(calls.length, 1);
});

test('chatStream: a caller abort propagates as AbortError, keeping deltas already delivered', async () => {
  const controller = new AbortController();
  const { fetch } = fakeFetch(stream([`data: ${JSON.stringify({ choices: [{ delta: { content: 'So far' } }] })}\n\n`], true));
  const deltas: string[] = [];
  const pending = client(fetch).chatStream({
    model: 'm',
    messages: [],
    signal: controller.signal,
    onDelta: (d) => {
      deltas.push(d);
      controller.abort();
    },
  });
  await assert.rejects(pending, (err: Error) => err.name === 'AbortError');
  assert.deepEqual(deltas, ['So far']);
});

test('timeouts become PROVIDER_TIMEOUT (retryable), both before and during streaming', async () => {
  const hangingFetch = fakeFetch((signal) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(signal.reason))));
  await rejectsWithCode(client(hangingFetch.fetch, { chat: 30 }).chat({ model: 'm', messages: [] }), 'PROVIDER_TIMEOUT', true);

  const stalledStream = fakeFetch(stream([`data: ${JSON.stringify({ choices: [{ delta: { content: 'a' } }] })}\n\n`], true));
  await rejectsWithCode(client(stalledStream.fetch, { chat: 30 }).chatStream({ model: 'm', messages: [], onDelta: () => {} }), 'PROVIDER_TIMEOUT', true);
});

test('HTTP and network failures map to typed errors, are retryable only when it makes sense, and are never retried', async () => {
  const tooMany = fakeFetch(json({ error: { message: 'Rate limit exceeded' } }, 429));
  await rejectsWithCode(client(tooMany.fetch).chat({ model: 'm', messages: [] }), 'PROVIDER_HTTP_429', true);
  assert.equal(tooMany.calls.length, 1);

  const serverError = fakeFetch(json({}, 503));
  await rejectsWithCode(client(serverError.fetch).chat({ model: 'm', messages: [] }), 'PROVIDER_HTTP_503', true);
  assert.equal(serverError.calls.length, 1);

  const badRequest = fakeFetch(json({ error: { message: 'Invalid model' } }, 400));
  await assert.rejects(client(badRequest.fetch).chat({ model: 'm', messages: [] }), (err: AppError) => {
    assert.equal(err.code, 'PROVIDER_HTTP_400');
    assert.equal(err.retryable, false);
    assert.match(err.message, /Invalid model/);
    return true;
  });

  const offline = fakeFetch(() => {
    throw new TypeError('fetch failed');
  });
  await rejectsWithCode(client(offline.fetch).embed({ model: 'e', input: ['x'] }), 'PROVIDER_UNREACHABLE', true);
  assert.equal(offline.calls.length, 1);

  const errorInBody = fakeFetch(json({ error: { message: 'No endpoints found' } }));
  await rejectsWithCode(client(errorInBody.fetch).chat({ model: 'm', messages: [] }), 'PROVIDER_ERROR', true);
});

const structuredArgs = {
  model: 'openai/gpt-6-astra',
  reasoningEffort: 'low' as const,
  schemaName: 'probe',
  jsonSchema: { type: 'object', additionalProperties: false, required: ['x'], properties: { x: { type: 'string' } } },
  messages: [{ role: 'user' as const, content: 'Extract x.' }],
  maxTokens: 32000,
  seed: 7,
};

const astraAnswer = (overrides: { content?: string | null; finish?: string; model?: string; refusal?: string | null } = {}) =>
  json({
    id: 'gen-9',
    model: overrides.model ?? 'openai/gpt-6-astra',
    provider: 'Azure',
    choices: [{ finish_reason: overrides.finish ?? 'stop', message: { content: overrides.content === undefined ? '{"x":"ok"}' : overrides.content, refusal: overrides.refusal ?? null } }],
    usage: { prompt_tokens: 80, completion_tokens: 57, cost: 0.00365, completion_tokens_details: { reasoning_tokens: 12 } },
  });

test('structured sends exactly the specified body and returns parsed JSON with usage', async () => {
  const { fetch, calls } = fakeFetch(astraAnswer());
  const result = await client(fetch).structured(structuredArgs);

  assert.deepEqual(calls[0]!.body, {
    model: 'openai/gpt-6-astra',
    reasoning: { effort: 'low', exclude: true },
    provider: { require_parameters: true },
    seed: 7,
    max_tokens: 32000,
    response_format: { type: 'json_schema', json_schema: { name: 'probe', strict: true, schema: structuredArgs.jsonSchema } },
    messages: structuredArgs.messages,
  });
  assert.ok(!('temperature' in calls[0]!.body) && !('top_p' in calls[0]!.body));
  assert.deepEqual(result.json, { x: 'ok' });
  assert.equal(result.model, 'openai/gpt-6-astra');
  assert.equal(result.provider, 'Azure');
  assert.equal(result.generationId, 'gen-9');
  assert.deepEqual(result.usage, { promptTokens: 80, completionTokens: 57, reasoningTokens: 12, cost: 0.00365 });
});

test('structured refuses incomplete, refused, filtered, mismatched, or non-JSON answers, without retrying', async () => {
  const cases: [Parameters<typeof astraAnswer>[0], string][] = [
    [{ refusal: 'I cannot help with that.' }, 'MODEL_REFUSED'],
    [{ finish: 'length', content: '{"x":"cut' }, 'OUTPUT_TRUNCATED'],
    [{ finish: 'content_filter' }, 'CONTENT_FILTERED'],
    [{ model: 'openai/gpt-4.1-mini' }, 'MODEL_MISMATCH'],
    [{ content: 'Sure! Here is the JSON: {' }, 'INVALID_JSON'],
  ];
  for (const [answer, code] of cases) {
    const { fetch, calls } = fakeFetch(astraAnswer(answer));
    await rejectsWithCode(client(fetch).structured(structuredArgs), code, false);
    assert.equal(calls.length, 1, `${code}: exactly one call`);
  }
});

test('embed returns vectors in input order, L2-normalized, with usage', async () => {
  const { fetch, calls } = fakeFetch(
    json({
      model: 'openai/text-embedding-3-small',
      data: [
        { index: 1, embedding: [0, 3, 4] },
        { index: 0, embedding: [2, 0, 0] },
      ],
      usage: { prompt_tokens: 2, cost: 4e-8 },
    }),
  );
  const result = await client(fetch).embed({ model: 'openai/text-embedding-3-small', input: ['first', 'second'] });
  assert.equal(calls[0]!.url, `${BASE}/embeddings`);
  assert.deepEqual(calls[0]!.body, { model: 'openai/text-embedding-3-small', input: ['first', 'second'] });
  assert.deepEqual([...result.vectors[0]!], [1, 0, 0]);
  assert.deepEqual([...result.vectors[1]!].map((v) => Math.round(v * 1000) / 1000), [0, 0.6, 0.8]);
  assert.equal(result.usage?.cost, 4e-8);
});

test('embed rejects missing, inconsistent, or invalid vectors; empty input makes no call', async () => {
  const cases: [unknown, string][] = [
    [{ data: [{ index: 0, embedding: [1, 0] }] }, 'PROVIDER_BAD_RESPONSE'], // 1 vector for 2 inputs
    [{ data: [{ index: 0, embedding: [1, 0] }, { index: 1, embedding: [1, 0, 0] }] }, 'PROVIDER_BAD_RESPONSE'],
    [{ data: [{ index: 0, embedding: [0, 0] }, { index: 1, embedding: [1, 0] }] }, 'PROVIDER_BAD_RESPONSE'],
    [{ data: [{ index: 0, embedding: ['a', 1] }, { index: 1, embedding: [1, 0] }] }, 'PROVIDER_BAD_RESPONSE'],
  ];
  for (const [body, code] of cases) {
    const { fetch } = fakeFetch(json(body));
    await rejectsWithCode(client(fetch).embed({ model: 'e', input: ['a', 'b'] }), code);
  }
  const { fetch, calls } = fakeFetch();
  assert.deepEqual((await client(fetch).embed({ model: 'e', input: [] })).vectors, []);
  assert.equal(calls.length, 0);
});

test('chat returns text and finish reason', async () => {
  const { fetch, calls } = fakeFetch(
    json({ id: 'gen-2', model: 'openai/gpt-4.1-mini', provider: 'OpenAI', choices: [{ finish_reason: 'stop', message: { content: 'OK' } }] }),
  );
  const result = await client(fetch).chat({ model: 'openai/gpt-4.1-mini', messages: [{ role: 'user', content: 'Reply with exactly OK.' }], maxTokens: 8 });
  assert.equal(result.text, 'OK');
  assert.equal(result.finishReason, 'stop');
  assert.equal(result.usage, null);
  assert.equal(calls[0]!.body.max_tokens, 8);
  assert.ok(!('stream' in calls[0]!.body));
});
