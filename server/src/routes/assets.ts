import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { Hono } from 'hono';
import { validator } from 'hono/validator';
import { IdParam } from '../../../shared/api.ts';
import type { AppDeps } from '../app.ts';
import { AppError } from '../errors.ts';
import { getAsset, uploadAsset } from '../services/assets.ts';
import { contentDisposition } from '../services/downloads.ts';
import { validate } from '../validate.ts';

export function videoAssetRoutes({ db, config }: AppDeps) {
  return new Hono().post('/:id/assets', validate('param', IdParam), validator('form', (body) => {
    if (!(body.file instanceof File)) throw new AppError(400, 'FILE_REQUIRED', 'Choose one file to upload.');
    return { file: body.file };
  }), async (c) => c.json(await uploadAsset(db, config.dataDir, c.req.valid('param').id, c.req.valid('form').file), 201));
}
export function assetRoutes({ db, config }: AppDeps) {
  return new Hono().get('/:id/content', validate('param', IdParam), async (c) => {
    const asset = getAsset(db, c.req.valid('param').id);
    const path = join(config.dataDir, 'files', asset.storageName);
    await stat(path).catch((err: NodeJS.ErrnoException) => { if (err.code === 'ENOENT') throw new AppError(404, 'NOT_FOUND', 'Attachment file not found.'); throw err; });
    const preview = asset.mediaType === 'application/pdf' || asset.mediaType.startsWith('image/');
    return new Response(Readable.toWeb(createReadStream(path)) as ReadableStream<Uint8Array>, { headers: {
      'Content-Type': asset.mediaType, 'Content-Length': String(asset.sizeBytes),
      'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': 'sandbox',
      'Content-Disposition': contentDisposition(preview && c.req.query('download') !== '1' ? 'inline' : 'attachment', asset.originalName),
    } });
  });
}
