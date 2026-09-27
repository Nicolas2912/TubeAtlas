// Saves a YouTube transcript as an evaluation snapshot: data/evaluation/<id>.transcript.json.
// Only for missing snapshots; an existing one is never overwritten (references depend on it).
// Usage: node scripts/fetch-transcript.ts <videoId or URL>
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { AppError } from '../server/src/errors.ts';
import { createYouTube, parseYouTubeId } from '../server/src/integrations/youtube.ts';

const arg = process.argv[2];
if (!arg) {
  console.error('Usage: node scripts/fetch-transcript.ts <videoId or URL>');
  process.exit(1);
}

try {
  const id = parseYouTubeId(arg);
  const path = `data/evaluation/${id}.transcript.json`;
  if (existsSync(path)) {
    console.error(`${path} already exists; not overwriting it.`);
    process.exit(1);
  }
  const result = await createYouTube({}).fetchTranscript(id);
  if (result.status !== 'ready') {
    console.error(`No transcript saved: ${result.status} (${result.message})`);
    process.exit(1);
  }
  // The existing snapshot format uses start + duration.
  const snapshot = {
    videoId: id,
    language: result.language,
    source: 'youtube_captions',
    segments: result.segments.map((s) => ({ id: s.id, start: s.start, duration: Math.round((s.end! - s.start!) * 1000) / 1000, text: s.text })),
  };
  const json = JSON.stringify(snapshot, null, 2);
  mkdirSync('data/evaluation', { recursive: true });
  writeFileSync(path, json);
  console.log(`Saved ${path}: ${snapshot.segments.length} segments (${result.language}), sha256 ${createHash('sha256').update(json).digest('hex')}`);
} catch (err) {
  console.error(err instanceof AppError ? `${err.code}: ${err.message}` : err);
  process.exit(1);
}
