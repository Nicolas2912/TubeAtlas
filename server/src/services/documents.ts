import { randomUUID } from 'node:crypto';
import { rename, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import type { z } from 'zod';
import type { CreateDocumentBody, UpdateDocumentBody } from '../../../shared/api.ts';
import { tx, type Db } from '../db.ts';
import { AppError } from '../errors.ts';
import { getVideo } from './videos.ts';

export type TextKind = 'note' | 'summary' | 'study_guide' | 'qa';
export type Asset = { id: number; originalName: string; mediaType: string; sizeBytes: number };
export type Document = { id: number; videoId: number; title: string; kind: TextKind | 'attachment'; markdown: string | null; asset: Asset | null; createdAt: string; updatedAt: string };
type Row = Omit<Document, 'asset'> & { assetId: number | null; originalName: string | null; mediaType: string | null; sizeBytes: number | null };
const select = `SELECT d.id, d.video_id AS videoId, d.title, d.kind, d.markdown, d.created_at AS createdAt, d.updated_at AS updatedAt,
  a.id AS assetId, a.original_name AS originalName, a.media_type AS mediaType, a.size_bytes AS sizeBytes
  FROM documents d LEFT JOIN assets a ON a.id = d.asset_id`;
function document(row: Row): Document {
  const { assetId, originalName, mediaType, sizeBytes, ...rest } = row;
  return { ...rest, asset: assetId === null ? null : { id: assetId, originalName: originalName!, mediaType: mediaType!, sizeBytes: sizeBytes! } };
}
export function getDocument(db: Db, id: number): Document {
  const row = db.prepare(`${select} WHERE d.id = ?`).get(id) as Row | undefined;
  if (!row) throw new AppError(404, 'NOT_FOUND', 'Document not found.');
  return document(row);
}
export function listDocuments(db: Db, videoId: number): Document[] {
  getVideo(db, videoId);
  return (db.prepare(`${select} WHERE d.video_id = ? ORDER BY d.updated_at DESC, d.id DESC`).all(videoId) as Row[]).map(document);
}
export function createDocument(db: Db, videoId: number, body: z.infer<typeof CreateDocumentBody>): Document {
  getVideo(db, videoId);
  const id = Number(db.prepare('INSERT INTO documents (video_id, title, kind, markdown) VALUES (?, ?, ?, ?)').run(videoId, body.title, body.kind, body.markdown).lastInsertRowid);
  return getDocument(db, id);
}
export function updateDocument(db: Db, id: number, body: z.infer<typeof UpdateDocumentBody>): Document {
  return tx(db, () => {
    const current = getDocument(db, id);
    if (current.asset && (body.kind !== undefined || body.markdown !== undefined || body.appendMarkdown !== undefined)) {
      throw new AppError(409, 'ATTACHMENT_READ_ONLY', 'Attachments can be renamed, but their content and kind cannot be edited.');
    }
    const markdown = body.appendMarkdown === undefined ? body.markdown ?? current.markdown : `${current.markdown}\n\n${body.appendMarkdown}`;
    if (markdown !== null && Buffer.byteLength(markdown) > 25 * 1024 * 1024) throw new AppError(413, 'DOCUMENT_TOO_LARGE', 'Notes must be 25 MB or smaller.');
    db.prepare(`UPDATE documents SET title = ?, kind = ?, markdown = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?`)
      .run(body.title ?? current.title, body.kind ?? current.kind, markdown, id);
    return getDocument(db, id);
  });
}
export async function deleteDocument(db: Db, dataDir: string, id: number) {
  const current = getDocument(db, id);
  let staged: { original: string; temporary: string } | undefined;
  if (current.asset) {
    const { storageName } = db.prepare('SELECT storage_name AS storageName FROM assets WHERE id = ?').get(current.asset.id) as { storageName: string };
    const original = join(dataDir, 'files', storageName);
    const temporary = join(dataDir, 'files', `tmp-delete-${randomUUID()}`);
    try { await rename(original, temporary); staged = { original, temporary }; }
    catch (err) { if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err; }
  }
  try {
    tx(db, () => {
      db.prepare('DELETE FROM documents WHERE id = ?').run(id);
      if (current.asset) db.prepare('DELETE FROM assets WHERE id = ?').run(current.asset.id);
    });
  } catch (err) { if (staged) await rename(staged.temporary, staged.original); throw err; }
  if (staged) await unlink(staged.temporary);
}
