// Opt-in live check of the OpenRouter integration: four tiny paid calls (well under $0.01 in total).
// Run with: npm run check:providers
import { z } from 'zod';
import { loadConfig } from '../server/src/config.ts';
import { AppError } from '../server/src/errors.ts';
import { createOpenRouter, type Usage } from '../server/src/integrations/openrouter.ts';

const config = loadConfig();
if (!config.openrouterApiKey) {
  console.error('OPENROUTER_API_KEY is not set (see .env.example).');
  process.exit(1);
}
const openrouter = createOpenRouter({ apiKey: config.openrouterApiKey, baseUrl: config.openrouterBaseUrl });

// Invented probe content, unrelated to any evaluation data.
const Book = z.strictObject({ title: z.string(), year: z.number().int(), author: z.string().nullable() });
const bookSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'year', 'author'],
  properties: { title: { type: 'string' }, year: { type: 'integer' }, author: { type: ['string', 'null'] } },
};

const checks: [string, () => Promise<{ model: string; provider?: string | null; usage: Usage | null; latencyMs?: number; detail: string }>][] = [
  [
    'chat',
    async () => {
      const r = await openrouter.chat({ model: config.chatModel, messages: [{ role: 'user', content: 'Reply with exactly OK.' }], maxTokens: 8 });
      if (!/^OK\.?$/i.test(r.text.trim())) throw new Error(`unexpected reply ${JSON.stringify(r.text)}`);
      return { ...r, detail: `reply ${JSON.stringify(r.text.trim())}` };
    },
  ],
  [
    'chatStream',
    async () => {
      let deltas = 0;
      const r = await openrouter.chatStream({
        model: config.chatModel,
        messages: [{ role: 'user', content: 'Reply with exactly OK.' }],
        maxTokens: 8,
        onDelta: () => deltas++,
      });
      if (!/^OK\.?$/i.test(r.text.trim())) throw new Error(`unexpected reply ${JSON.stringify(r.text)}`);
      return { ...r, detail: `reply ${JSON.stringify(r.text.trim())} in ${deltas} delta(s)` };
    },
  ],
  [
    'embed',
    async () => {
      const started = Date.now();
      const r = await openrouter.embed({ model: config.embeddingModel, input: ['A bicycle has two wheels.', 'Bread needs flour.'] });
      const norms = r.vectors.map((v) => Math.sqrt(v.reduce((sum, x) => sum + x * x, 0)).toFixed(3));
      return { ...r, latencyMs: Date.now() - started, detail: `${r.vectors.length} vectors × ${r.vectors[0]!.length} dims, norms ${norms.join('/')}` };
    },
  ],
  [
    'structured',
    async () => {
      const r = await openrouter.structured({
        model: config.kgModel,
        reasoningEffort: config.kgReasoningEffort,
        schemaName: 'book_probe',
        jsonSchema: bookSchema,
        messages: [{ role: 'user', content: 'Extract the book: "The Lighthouse Keeper\'s Almanac was first printed in 1897; its author is unknown."' }],
        maxTokens: 400,
        seed: 7,
      });
      const book = Book.parse(r.json);
      return {
        ...r,
        detail: `effort ${config.kgReasoningEffort}, reasoning tokens ${r.usage?.reasoningTokens ?? 'n/a'}, ${JSON.stringify(book)}`,
      };
    },
  ],
];

let failed = 0;
let totalCost = 0;
for (const [name, run] of checks) {
  try {
    const r = await run();
    const cost = r.usage?.cost;
    if (typeof cost === 'number') totalCost += cost;
    console.log(
      `PASS ${name.padEnd(10)} ${r.model} via ${r.provider ?? 'n/a'}, ${r.latencyMs ?? '?'} ms, cost ${cost === null || cost === undefined ? 'not reported' : `$${cost.toFixed(6)}`} — ${r.detail}`,
    );
  } catch (err) {
    failed++;
    const reason = err instanceof AppError ? `${err.code}: ${err.message}` : err instanceof Error ? err.message : String(err);
    console.log(`FAIL ${name.padEnd(10)} ${reason}`);
  }
}
console.log(`${checks.length - failed}/${checks.length} passed; reported cost $${totalCost.toFixed(6)}`);
process.exitCode = failed ? 1 : 0;
