import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AppError } from '../errors.ts';
import { cleanText, createYouTube, parseIsoDuration, parseYouTubeId, toSegments } from './youtube.ts';

const ID = 'AbCdEfGhIjK';

test('parseYouTubeId accepts the common link forms and bare ids', () => {
  const forms = [
    ID,
    ` ${ID} `,
    `https://www.youtube.com/watch?v=${ID}`,
    `https://www.youtube.com/watch?feature=share&v=${ID}&t=42s`,
    `youtube.com/watch?v=${ID}`,
    `https://m.youtube.com/watch?v=${ID}`,
    `https://music.youtube.com/watch?v=${ID}&list=x`,
    `https://youtu.be/${ID}`,
    `https://youtu.be/${ID}?si=abc&t=10`,
    `https://www.youtube.com/shorts/${ID}`,
    `https://www.youtube.com/embed/${ID}?start=5`,
    `https://www.youtube-nocookie.com/embed/${ID}`,
    `https://www.youtube.com/live/${ID}?feature=share`,
  ];
  for (const form of forms) assert.equal(parseYouTubeId(form), ID, form);
});

test('parseYouTubeId rejects everything else with 400 INVALID_URL', () => {
  const rejects = [
    '',
    'hello world',
    'AbCdEfGhIj', // 10 characters
    'https://vimeo.com/123456789',
    `https://example.com/watch?v=${ID}`,
    'https://www.youtube.com/watch?v=short',
    'https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv',
    'https://www.youtube.com/playlist?list=PL123',
    `https://www.youtube.com.evil.example/watch?v=${ID}`,
  ];
  for (const input of rejects) {
    assert.throws(() => parseYouTubeId(input), (err: AppError) => err.code === 'INVALID_URL' && err.status === 400, input);
  }
});

test('parseIsoDuration handles hours, minutes, seconds, days, and live (P0D)', () => {
  assert.equal(parseIsoDuration('PT9M50S'), 590);
  assert.equal(parseIsoDuration('PT2H39M53S'), 9593);
  assert.equal(parseIsoDuration('PT45S'), 45);
  assert.equal(parseIsoDuration('P1DT1H'), 90000);
  assert.equal(parseIsoDuration('P0D'), null);
  assert.equal(parseIsoDuration('nonsense'), null);
});

test('toSegments converts srv3 milliseconds and classic seconds to seconds, dropping empty text', () => {
  assert.deepEqual(
    toSegments(
      [
        { text: 'First words', offset: 1920, duration: 4320 },
        { text: '   ', offset: 3000, duration: 1000 },
        { text: 'next', offset: 6240, duration: 2000 },
      ],
      'srv3',
    ),
    [
      { id: 0, start: 1.92, end: 6.24, text: 'First words' },
      { id: 1, start: 6.24, end: 8.24, text: 'next' },
    ],
  );
  assert.deepEqual(toSegments([{ text: 'Hello', offset: 1.5, duration: 2.25 }], 'classic'), [{ id: 0, start: 1.5, end: 3.75, text: 'Hello' }]);
});

test('cleanText decodes leftover entities and collapses whitespace', () => {
  assert.equal(cleanText('It&amp;#39;s&nbsp; a\n  test &gt; ok'), "It's a test > ok");
});

// ---- adapter with a fake network ----

type Route = (url: string, init?: RequestInit) => Response | Promise<Response>;

function fakeNetwork(routes: Record<string, Route>) {
  const calls: string[] = [];
  const fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    calls.push(url);
    if (init?.signal?.aborted) throw init.signal.reason;
    const key = Object.keys(routes).find((prefix) => url.startsWith(prefix));
    if (!key) throw new TypeError(`fetch failed (no route for ${url})`);
    return routes[key]!(url, init);
  }) as typeof globalThis.fetch;
  return { fetch, calls };
}

const PLAYER = 'https://www.youtube.com/youtubei/v1/player';
const CAPTIONS = 'https://www.youtube.com/api/timedtext';
const WATCH = 'https://www.youtube.com/watch';
const tracks = (lang = 'en') => () =>
  Response.json({ captions: { playerCaptionsTracklistRenderer: { captionTracks: [{ baseUrl: `${CAPTIONS}?v=${ID}&lang=${lang}`, languageCode: lang }] } } });
const xml = (body: string, status = 200) => () => new Response(body, { status, headers: { 'content-type': 'text/xml' } });

// Invented caption content.
const SRV3 = `<?xml version="1.0" encoding="utf-8" ?><timedtext format="3"><body>
<p t="1920" d="4320"><s>The bakery</s><s> opens early</s></p>
<p t="6240" d="3100">Fresh bread &amp;amp; rolls</p>
<p t="9000" d="500"></p>
</body></timedtext>`;
const CLASSIC = `<?xml version="1.0" encoding="utf-8" ?><transcript>
<text start="0.5" dur="2.25">Welcome to the garden</text>
<text start="2.75" dur="3">Tomatoes need sun</text>
</transcript>`;

test('fetchTranscript: srv3 captions arrive in milliseconds and are stored in seconds', async () => {
  const { fetch } = fakeNetwork({ [PLAYER]: tracks('de'), [CAPTIONS]: xml(SRV3) });
  const result = await createYouTube({ fetch }).fetchTranscript(ID);
  assert.deepEqual(result, {
    status: 'ready',
    language: 'de',
    segments: [
      { id: 0, start: 1.92, end: 6.24, text: 'The bakery opens early' },
      { id: 1, start: 6.24, end: 9.34, text: 'Fresh bread & rolls' },
    ],
  });
});

test('fetchTranscript: classic captions arrive in seconds', async () => {
  const { fetch } = fakeNetwork({ [PLAYER]: tracks(), [CAPTIONS]: xml(CLASSIC) });
  const result = await createYouTube({ fetch }).fetchTranscript(ID);
  assert.equal(result.status, 'ready');
  assert.deepEqual(result.status === 'ready' && result.segments, [
    { id: 0, start: 0.5, end: 2.75, text: 'Welcome to the garden' },
    { id: 1, start: 2.75, end: 5.75, text: 'Tomatoes need sun' },
  ]);
});

test('fetchTranscript: an unrecognized caption format fails loudly instead of guessing units', async () => {
  const { fetch } = fakeNetwork({ [PLAYER]: tracks(), [CAPTIONS]: xml('<timedtext><event start="1">x</event></timedtext>') });
  await assert.rejects(createYouTube({ fetch }).fetchTranscript(ID), (err: AppError) => err.code === 'UNKNOWN_CAPTION_FORMAT');
});

test('fetchTranscript: 429 on the caption file means blocked, not missing', async () => {
  const { fetch } = fakeNetwork({ [PLAYER]: tracks(), [CAPTIONS]: xml('Too Many Requests', 429) });
  assert.equal((await createYouTube({ fetch }).fetchTranscript(ID)).status, 'blocked');
});

test('fetchTranscript: a captcha page means blocked', async () => {
  const { fetch } = fakeNetwork({
    [PLAYER]: () => Response.json({}),
    [WATCH]: () => new Response('<html><div class="g-recaptcha"></div></html>'),
  });
  assert.equal((await createYouTube({ fetch }).fetchTranscript(ID)).status, 'blocked');
});

test('fetchTranscript: a playable video without caption tracks means no_captions', async () => {
  const { fetch } = fakeNetwork({
    [PLAYER]: () => Response.json({}),
    [WATCH]: () => new Response('<html><script>var ytInitialPlayerResponse = {"playabilityStatus":{"status":"OK"}};</script></html>'),
  });
  assert.deepEqual(await createYouTube({ fetch }).fetchTranscript(ID), { status: 'no_captions', message: 'This video has no captions.' });
});

test('fetchTranscript: an unavailable video fails with a clear message', async () => {
  const { fetch } = fakeNetwork({ [PLAYER]: () => Response.json({}), [WATCH]: () => new Response('<html>gone</html>') });
  assert.deepEqual(await createYouTube({ fetch }).fetchTranscript(ID), { status: 'failed', message: 'Video unavailable' });
});

test('fetchTranscript: network errors are retried twice (1 s, 3 s), then reported as failed', async () => {
  const delays: number[] = [];
  const { fetch, calls } = fakeNetwork({}); // every request fails at the network level
  const result = await createYouTube({ fetch, sleep: async (ms) => void delays.push(ms) }).fetchTranscript(ID);
  assert.equal(result.status, 'failed');
  assert.deepEqual(delays, [1000, 3000]);
  assert.equal(calls.filter((url) => url.startsWith(WATCH)).length, 3);
});

test('fetchTranscript: a network error followed by success returns the transcript', async () => {
  let attempts = 0;
  const { fetch } = fakeNetwork({
    [PLAYER]: () => {
      attempts++;
      if (attempts === 1) throw new TypeError('fetch failed');
      return tracks()();
    },
    [WATCH]: () => {
      throw new TypeError('fetch failed');
    },
    [CAPTIONS]: xml(CLASSIC),
  });
  const result = await createYouTube({ fetch, sleep: async () => {} }).fetchTranscript(ID);
  assert.equal(result.status, 'ready');
});

test('fetchTranscript: a caller abort propagates and is not retried', async () => {
  const controller = new AbortController();
  controller.abort();
  const { fetch } = fakeNetwork({ [PLAYER]: tracks(), [CAPTIONS]: xml(CLASSIC) });
  await assert.rejects(createYouTube({ fetch, sleep: async () => assert.fail('must not retry') }).fetchTranscript(ID, controller.signal), (err: Error) => err.name === 'AbortError');
});

// ---- metadata ----

const DATA_API = 'https://www.googleapis.com/youtube/v3/videos';
const OEMBED = 'https://www.youtube.com/oembed';

test('fetchMetadata with a key uses the Data API, including duration and the best thumbnail', async () => {
  const { fetch, calls } = fakeNetwork({
    [DATA_API]: () =>
      Response.json({
        items: [
          {
            snippet: { title: 'Garden basics', channelTitle: 'Green Hands', thumbnails: { default: { url: 'd.jpg' }, high: { url: 'h.jpg' } } },
            contentDetails: { duration: 'PT10M38S' },
          },
        ],
      }),
  });
  assert.deepEqual(await createYouTube({ apiKey: 'k', fetch }).fetchMetadata(ID), {
    youtubeId: ID,
    title: 'Garden basics',
    channel: 'Green Hands',
    durationSeconds: 638,
    thumbnailUrl: 'h.jpg',
  });
  assert.equal(calls.length, 1);
});

test('fetchMetadata: no Data API result means 404 VIDEO_NOT_FOUND', async () => {
  const { fetch } = fakeNetwork({ [DATA_API]: () => Response.json({ items: [] }) });
  await assert.rejects(createYouTube({ apiKey: 'k', fetch }).fetchMetadata(ID), (err: AppError) => err.code === 'VIDEO_NOT_FOUND' && err.status === 404);
});

test('fetchMetadata falls back to oEmbed when the Data API rejects the key or quota', async (t) => {
  t.mock.method(console, 'warn', () => {});
  const { fetch, calls } = fakeNetwork({
    [DATA_API]: () => Response.json({ error: { message: 'quota' } }, { status: 403 }),
    [OEMBED]: () => Response.json({ title: 'Garden basics', author_name: 'Green Hands', thumbnail_url: 'o.jpg' }),
  });
  const meta = await createYouTube({ apiKey: 'k', fetch }).fetchMetadata(ID);
  assert.deepEqual(meta, { youtubeId: ID, title: 'Garden basics', channel: 'Green Hands', durationSeconds: null, thumbnailUrl: 'o.jpg' });
  assert.equal(calls.length, 2);
});

test('fetchMetadata without a key uses oEmbed; 404, 401, and network errors are typed', async () => {
  const cases: [Route | 'offline', string][] = [
    [() => new Response('Not Found', { status: 404 }), 'VIDEO_NOT_FOUND'],
    [() => new Response('Unauthorized', { status: 401 }), 'VIDEO_RESTRICTED'],
    ['offline', 'YOUTUBE_UNREACHABLE'],
  ];
  for (const [route, code] of cases) {
    const { fetch } = fakeNetwork(route === 'offline' ? {} : { [OEMBED]: route });
    await assert.rejects(createYouTube({ fetch }).fetchMetadata(ID), (err: AppError) => err.code === code, code);
  }
});
