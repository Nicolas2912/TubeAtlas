// YouTube metadata (Data API or oEmbed) and captions (unofficial youtube-transcript package).
import {
  YoutubeTranscript,
  YoutubeTranscriptDisabledError,
  YoutubeTranscriptNotAvailableError,
  YoutubeTranscriptTooManyRequestError,
  YoutubeTranscriptVideoUnavailableError,
} from 'youtube-transcript';
import type { Segment, TranscriptStatus } from '../../../shared/api.ts';
import { AppError } from '../errors.ts';

export type VideoMetadata = {
  youtubeId: string;
  title: string;
  channel: string | null;
  durationSeconds: number | null;
  thumbnailUrl: string | null;
};

export type TranscriptFetchResult =
  | { status: 'ready'; language: string | null; segments: Segment[] }
  | { status: Exclude<TranscriptStatus, 'ready' | 'pending'>; message: string };

export type YouTube = {
  fetchMetadata(youtubeId: string): Promise<VideoMetadata>;
  fetchTranscript(youtubeId: string, signal?: AbortSignal): Promise<TranscriptFetchResult>;
};

const ID = /^[A-Za-z0-9_-]{11}$/;
const METADATA_TIMEOUT_MS = 15_000;
const CAPTION_REQUEST_TIMEOUT_MS = 30_000;

/** Extracts the 11-character video id from a URL or bare id; 400 INVALID_URL otherwise. */
export function parseYouTubeId(input: string): string {
  const value = input.trim();
  if (ID.test(value)) return value;
  let url: URL;
  try {
    url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(value) ? value : `https://${value}`);
  } catch {
    throw invalidUrl();
  }
  const host = url.hostname.toLowerCase().replace(/^(www\.|m\.|music\.)/, '');
  let candidate: string | null | undefined;
  if (host === 'youtu.be') candidate = url.pathname.split('/')[1];
  else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
    const [first, second] = url.pathname.split('/').filter(Boolean);
    if (first === 'watch') candidate = url.searchParams.get('v');
    else if (first && ['shorts', 'embed', 'live', 'v', 'e'].includes(first)) candidate = second;
  }
  if (!candidate || !ID.test(candidate)) throw invalidUrl();
  return candidate;
}

const invalidUrl = () => new AppError(400, 'INVALID_URL', 'That is not a YouTube video link.');

/** ISO 8601 durations as used by the YouTube Data API, e.g. PT1H2M3S or P1DT2H. */
export function parseIsoDuration(value: string): number | null {
  const m = /^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$/.exec(value);
  if (!m) return null;
  const [, d = '0', h = '0', min = '0', s = '0'] = m;
  const total = Number(d) * 86400 + Number(h) * 3600 + Number(min) * 60 + Number(s);
  return total > 0 ? total : null; // P0D: live or upcoming
}

/**
 * Converts the package's segments to seconds. The unit depends on the caption format the package
 * parsed: srv3 (<p t="…">) reports milliseconds, the classic format (<text start="…">) seconds.
 */
export function toSegments(raw: { text: string; offset: number; duration: number }[], format: 'srv3' | 'classic'): Segment[] {
  const scale = format === 'srv3' ? 1000 : 1;
  const segments: Segment[] = [];
  for (const item of raw) {
    const text = cleanText(item.text);
    if (!text) continue;
    const start = round3(item.offset / scale);
    segments.push({ id: segments.length, start, end: round3(start + item.duration / scale), text });
  }
  return segments;
}

/** Decodes entities the package leaves behind (e.g. double-encoded or &nbsp;) and collapses whitespace. */
export function cleanText(text: string): string {
  return text
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex: string) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec: string) => String.fromCodePoint(parseInt(dec, 10)))
    .replace(/\s+/g, ' ')
    .trim();
}

const round3 = (n: number) => Math.round(n * 1000) / 1000;

export function createYouTube(options: {
  apiKey?: string;
  fetch?: typeof globalThis.fetch;
  /** Delay before retrying after a network error; overridable for tests. */
  sleep?: (ms: number) => Promise<void>;
}): YouTube {
  const fetchFn = options.fetch ?? globalThis.fetch;
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));

  async function getJson(url: string): Promise<{ status: number; body: any }> {
    let res: Response;
    try {
      res = await fetchFn(url, { signal: AbortSignal.timeout(METADATA_TIMEOUT_MS) });
    } catch {
      throw new AppError(502, 'YOUTUBE_UNREACHABLE', 'Could not reach YouTube.', true);
    }
    const text = await res.text().catch(() => '');
    let body: any = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    return { status: res.status, body };
  }

  async function fromDataApi(id: string, key: string): Promise<VideoMetadata | 'fallback'> {
    const url = `https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails&id=${id}&key=${encodeURIComponent(key)}`;
    const { status, body } = await getJson(url);
    if (status !== 200) {
      // Bad key or exhausted quota: oEmbed still gives title, channel, and thumbnail.
      console.warn(`YouTube Data API returned HTTP ${status}; falling back to oEmbed.`);
      return 'fallback';
    }
    const item = body?.items?.[0];
    if (!item) throw new AppError(404, 'VIDEO_NOT_FOUND', 'YouTube has no video with that id.');
    const thumbs = item.snippet?.thumbnails ?? {};
    const thumb = thumbs.maxres ?? thumbs.standard ?? thumbs.high ?? thumbs.medium ?? thumbs.default;
    return {
      youtubeId: id,
      title: String(item.snippet?.title ?? id),
      channel: item.snippet?.channelTitle ?? null,
      durationSeconds: typeof item.contentDetails?.duration === 'string' ? parseIsoDuration(item.contentDetails.duration) : null,
      thumbnailUrl: thumb?.url ?? null,
    };
  }

  async function fromOEmbed(id: string): Promise<VideoMetadata> {
    const watchUrl = `https://www.youtube.com/watch?v=${id}`;
    const { status, body } = await getJson(`https://www.youtube.com/oembed?url=${encodeURIComponent(watchUrl)}&format=json`);
    if (status === 404 || status === 400) throw new AppError(404, 'VIDEO_NOT_FOUND', 'YouTube has no video with that id.');
    if (status === 401 || status === 403) {
      throw new AppError(
        422,
        'VIDEO_RESTRICTED',
        'YouTube does not share details for this video (private, or embedding disabled). Adding a YOUTUBE_API_KEY may help.',
      );
    }
    if (status !== 200 || !body?.title) throw new AppError(502, 'YOUTUBE_UNREACHABLE', `YouTube returned HTTP ${status}.`, true);
    return { youtubeId: id, title: String(body.title), channel: body.author_name ?? null, durationSeconds: null, thumbnailUrl: body.thumbnail_url ?? null };
  }

  async function fetchMetadata(id: string): Promise<VideoMetadata> {
    if (options.apiKey) {
      const result = await fromDataApi(id, options.apiKey);
      if (result !== 'fallback') return result;
    }
    return fromOEmbed(id);
  }

  async function fetchTranscriptOnce(id: string, signal?: AbortSignal): Promise<TranscriptFetchResult> {
    let format: 'srv3' | 'classic' | 'unknown' | null = null;
    let sawTooManyRequests = false;
    // Wraps every request the package makes: adds a timeout and the caller's signal, and records
    // the caption format (it decides the time unit) and any 429 (YouTube blocking this IP).
    const recordingFetch = (async (input: string | URL | Request, init?: RequestInit) => {
      const timeout = AbortSignal.timeout(CAPTION_REQUEST_TIMEOUT_MS);
      const res = await fetchFn(input, { ...init, signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
      if (res.status === 429) sawTooManyRequests = true;
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (url.includes('/api/timedtext') && res.ok) {
        const body = await res.clone().text();
        format = /<p\s+t="/.test(body) ? 'srv3' : /<text\s+start="/.test(body) ? 'classic' : 'unknown';
      }
      return res;
    }) as typeof globalThis.fetch;

    let raw: Awaited<ReturnType<typeof YoutubeTranscript.fetchTranscript>>;
    try {
      raw = await YoutubeTranscript.fetchTranscript(id, { fetch: recordingFetch });
    } catch (err) {
      if (signal?.aborted) throw err;
      if (sawTooManyRequests || err instanceof YoutubeTranscriptTooManyRequestError) {
        return { status: 'blocked', message: 'YouTube is temporarily blocking caption requests from this network. Try again later.' };
      }
      if (err instanceof YoutubeTranscriptDisabledError || err instanceof YoutubeTranscriptNotAvailableError) {
        return { status: 'no_captions', message: 'This video has no captions.' };
      }
      if (err instanceof YoutubeTranscriptVideoUnavailableError) return { status: 'failed', message: 'Video unavailable' };
      throw err; // network and unexpected errors: the caller decides about retrying
    }
    // Check the format first: a format the package can't read also comes back as an empty list.
    if (format === 'unknown' || (format === null && raw.length > 0)) {
      throw new AppError(502, 'UNKNOWN_CAPTION_FORMAT', 'YouTube returned captions in an unrecognized format.', false);
    }
    if (raw.length === 0 || format === null) return { status: 'no_captions', message: 'This video has no captions.' };
    const segments = toSegments(raw, format);
    if (segments.length === 0) return { status: 'no_captions', message: 'This video has no captions.' };
    return { status: 'ready', language: raw[0]?.lang ?? null, segments };
  }

  /** Fetches captions; network failures are retried twice (1 s, 3 s) because caption requests are free. */
  async function fetchTranscript(id: string, signal?: AbortSignal): Promise<TranscriptFetchResult> {
    const delays = [1000, 3000];
    for (let attempt = 0; ; attempt++) {
      try {
        return await fetchTranscriptOnce(id, signal);
      } catch (err) {
        if (signal?.aborted || err instanceof AppError) throw err;
        if (attempt >= delays.length) {
          return { status: 'failed', message: 'Could not reach YouTube to fetch captions. Try again later.' };
        }
        await sleep(delays[attempt]!);
      }
    }
  }

  return { fetchMetadata, fetchTranscript };
}
