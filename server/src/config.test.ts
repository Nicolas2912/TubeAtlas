import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isAbsolute } from 'node:path';
import { loadConfig } from './config.ts';

test('defaults match .env.example and empty values count as unset', () => {
  const config = loadConfig({ OPENROUTER_API_KEY: '', YOUTUBE_API_KEY: '  ' });
  assert.equal(config.openrouterApiKey, undefined);
  assert.equal(config.youtubeApiKey, undefined);
  assert.equal(config.openrouterBaseUrl, 'https://openrouter.ai/api/v1');
  assert.equal(config.chatModel, 'openai/gpt-4.1-mini');
  assert.equal(config.embeddingModel, 'openai/text-embedding-3-small');
  assert.equal(config.kgModel, 'openai/gpt-6-astra');
  assert.equal(config.kgReasoningEffort, 'low');
  assert.equal(config.port, 5170);
  assert.ok(isAbsolute(config.dataDir));
  assert.ok(Object.isFrozen(config));
});

test('GOOGLE_API_KEY is an alias; YOUTUBE_API_KEY wins', () => {
  assert.equal(loadConfig({ GOOGLE_API_KEY: 'g' }).youtubeApiKey, 'g');
  assert.equal(loadConfig({ GOOGLE_API_KEY: 'g', YOUTUBE_API_KEY: 'y' }).youtubeApiKey, 'y');
});

test('invalid values fail with variable names but never values', () => {
  assert.throws(
    () => loadConfig({ PORT: 'abc', OPENROUTER_KG_REASONING_EFFORT: 'extreme', OPENROUTER_API_KEY: 'sk-or-secret' }),
    (err: Error) => err.message.includes('PORT') && err.message.includes('OPENROUTER_KG_REASONING_EFFORT') && !err.message.includes('secret'),
  );
});
