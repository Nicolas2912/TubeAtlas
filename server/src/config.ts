import { resolve } from 'node:path';
import { z } from 'zod';

// Empty values in .env (e.g. `YOUTUBE_API_KEY=`) mean "not set".
const optional = z.preprocess((v) => (typeof v === 'string' && v.trim() === '' ? undefined : v), z.string().trim().optional());
const withDefault = (fallback: string) =>
  z.preprocess((v) => (typeof v === 'string' && v.trim() === '' ? undefined : v), z.string().trim().default(fallback));

const Env = z.object({
  OPENROUTER_API_KEY: optional,
  OPENROUTER_BASE_URL: withDefault('https://openrouter.ai/api/v1').pipe(z.url()),
  OPENROUTER_CHAT_MODEL: withDefault('openai/gpt-4.1-mini'),
  OPENROUTER_EMBEDDING_MODEL: withDefault('openai/text-embedding-3-small'),
  OPENROUTER_KG_MODEL: withDefault('openai/gpt-6-astra'),
  OPENROUTER_KG_REASONING_EFFORT: withDefault('low').pipe(z.enum(['minimal', 'low', 'medium', 'high'])),
  YOUTUBE_API_KEY: optional,
  GOOGLE_API_KEY: optional,
  DATA_DIR: withDefault('./data'),
  PORT: withDefault('5170').pipe(z.string().regex(/^\d+$/)).transform(Number).pipe(z.number().int().min(1).max(65535)),
});

export type Config = Readonly<{
  openrouterApiKey: string | undefined;
  openrouterBaseUrl: string;
  chatModel: string;
  embeddingModel: string;
  kgModel: string;
  kgReasoningEffort: 'minimal' | 'low' | 'medium' | 'high';
  youtubeApiKey: string | undefined;
  dataDir: string;
  port: number;
}>;

/** Reads configuration from environment variables. Error messages name variables, never values. */
export function loadConfig(env: Record<string, string | undefined> = process.env): Config {
  const parsed = Env.safeParse(env);
  if (!parsed.success) {
    const names = [...new Set(parsed.error.issues.map((issue) => String(issue.path[0])))];
    throw new Error(`Invalid configuration: ${names.join(', ')}. See .env.example.`);
  }
  const e = parsed.data;
  return Object.freeze({
    openrouterApiKey: e.OPENROUTER_API_KEY,
    openrouterBaseUrl: e.OPENROUTER_BASE_URL.replace(/\/+$/, ''),
    chatModel: e.OPENROUTER_CHAT_MODEL,
    embeddingModel: e.OPENROUTER_EMBEDDING_MODEL,
    kgModel: e.OPENROUTER_KG_MODEL,
    kgReasoningEffort: e.OPENROUTER_KG_REASONING_EFFORT,
    youtubeApiKey: e.YOUTUBE_API_KEY ?? e.GOOGLE_API_KEY,
    dataDir: resolve(e.DATA_DIR),
    port: e.PORT,
  });
}
