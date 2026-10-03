import { Hono } from 'hono';
import { CreateDocumentBody, IdParam, UpdateDocumentBody } from '../../../shared/api.ts';
import type { AppDeps } from '../app.ts';
import { AppError } from '../errors.ts';
import { createDocument, deleteDocument, getDocument, listDocuments, updateDocument } from '../services/documents.ts';
import { contentDisposition } from '../services/downloads.ts';
import { validate } from '../validate.ts';

export function videoDocumentRoutes({ db }: AppDeps) {
  return new Hono()
    .get('/:id/documents', validate('param', IdParam), (c) => c.json(listDocuments(db, c.req.valid('param').id)))
    .post('/:id/documents', validate('param', IdParam), validate('json', CreateDocumentBody), (c) => c.json(createDocument(db, c.req.valid('param').id, c.req.valid('json')), 201));
}
export function documentRoutes({ db, config }: AppDeps) {
  return new Hono()
    .get('/:id', validate('param', IdParam), (c) => c.json(getDocument(db, c.req.valid('param').id)))
    .patch('/:id', validate('param', IdParam), validate('json', UpdateDocumentBody), (c) => c.json(updateDocument(db, c.req.valid('param').id, c.req.valid('json'))))
    .delete('/:id', validate('param', IdParam), async (c) => { await deleteDocument(db, config.dataDir, c.req.valid('param').id); return c.body(null, 204); })
    .get('/:id/export', validate('param', IdParam), (c) => {
      const doc = getDocument(db, c.req.valid('param').id);
      if (doc.markdown === null) throw new AppError(409, 'ATTACHMENT_READ_ONLY', 'Download this attachment in its original format.');
      c.header('Content-Type', 'text/markdown; charset=utf-8');
      c.header('Content-Disposition', contentDisposition('attachment', `${doc.title}.md`));
      c.header('X-Content-Type-Options', 'nosniff');
      return c.body(doc.markdown);
    });
}
