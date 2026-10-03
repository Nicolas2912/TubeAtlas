import { randomUUID } from 'node:crypto';
import { rename, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tx, type Db } from '../db.ts';
import { AppError } from '../errors.ts';
import { createDocument, getDocument, type Document } from './documents.ts';
import { getVideo } from './videos.ts';

const types: Record<string, string> = { pdf: 'application/pdf', png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' };
function supported(bytes: Buffer, extension: string) {
  if (extension === 'pdf') return bytes.subarray(0, 4).toString() === '%PDF';
  if (extension === 'png') return bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  if (extension === 'jpg' || extension === 'jpeg') return bytes.subarray(0, 3).equals(Buffer.from([255, 216, 255]));
  if (extension === 'webp') return bytes.subarray(0, 4).toString() === 'RIFF' && bytes.subarray(8, 12).toString() === 'WEBP';
  return ['docx', 'xlsx', 'pptx'].includes(extension) && bytes.subarray(0, 4).equals(Buffer.from([80, 75, 3, 4]));
}
export async function uploadAsset(db: Db, dataDir: string, videoId: number, file: File): Promise<Document> {
  getVideo(db, videoId);
  if (file.size > 25 * 1024 * 1024) throw new AppError(413, 'PAYLOAD_TOO_LARGE', 'Files must be 25 MB or smaller.');
  const originalName = file.name.split(/[/\\]/).pop()!.replace(/[\u0000-\u001f\u007f]/g, '_').slice(0, 200) || 'file';
  const extension = originalName.split('.').pop()!.toLowerCase();
  const bytes = Buffer.from(await file.arrayBuffer());
  if (extension === 'md' || extension === 'txt') {
    let text: string;
    try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); if (text.includes('\0')) throw new Error('binary'); }
    catch { throw new AppError(415, 'UNSUPPORTED_FILE', 'Text files must contain valid UTF-8 text.'); }
    return createDocument(db, videoId, { title: originalName, kind: 'note', markdown: text });
  }
  if (!types[extension] || !supported(bytes, extension)) throw new AppError(415, 'UNSUPPORTED_FILE', 'Choose a PDF, PNG, JPEG, WebP, Office, Markdown, or text file matching its extension.');
  const uuid = randomUUID();
  const temporary = join(dataDir, 'files', `tmp-${uuid}`);
  const storageName = `${uuid}.${extension}`;
  const target = join(dataDir, 'files', storageName);
  try {
    await writeFile(temporary, bytes, { flag: 'wx' });
    await rename(temporary, target);
    return tx(db, () => {
      const assetId = Number(db.prepare('INSERT INTO assets (video_id, storage_name, original_name, media_type, size_bytes) VALUES (?, ?, ?, ?, ?)')
        .run(videoId, storageName, originalName, types[extension], bytes.length).lastInsertRowid);
      const documentId = Number(db.prepare("INSERT INTO documents (video_id, title, kind, asset_id) VALUES (?, ?, 'attachment', ?)").run(videoId, originalName, assetId).lastInsertRowid);
      return getDocument(db, documentId);
    });
  } catch (err) {
    await Promise.all([temporary, target].map((path) => unlink(path).catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error; })));
    throw err;
  }
}
export function getAsset(db: Db, id: number) {
  const asset = db.prepare('SELECT storage_name AS storageName, original_name AS originalName, media_type AS mediaType, size_bytes AS sizeBytes FROM assets WHERE id = ?')
    .get(id) as { storageName: string; originalName: string; mediaType: string; sizeBytes: number } | undefined;
  if (!asset) throw new AppError(404, 'NOT_FOUND', 'Attachment not found.');
  return asset;
}
