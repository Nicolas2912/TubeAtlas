import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createTestApp } from '../testing.ts';

function setup(t: { after(fn: () => Promise<void>): void }) {
  const app = createTestApp();
  t.after(app.cleanup);
  app.db.prepare("INSERT INTO videos (youtube_id, title) VALUES ('AbCdEfGhIjK', 'Paper lanterns'), ('LmNoPqRsTuV', 'Other workshop')").run();
  const upload = (name: string, bytes: Uint8Array | string, videoId = 1) => {
    const form = new FormData();
    form.append('file', new File([typeof bytes === 'string' ? bytes : Uint8Array.from(bytes)], name, { type: 'text/html' }));
    return app.request(`/api/videos/${videoId}/assets`, { method: 'POST', body: form });
  };
  return { ...app, upload };
}
const sha = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

test('documents persist CRUD, validate edits, scope lists, append without replacing, and export safely', async (t) => {
  const { json, request } = setup(t);
  const response = await json('POST', '/api/videos/1/documents', { title: 'Workshop notes', markdown: 'First fold.' });
  assert.equal(response.status, 201);
  const doc = await response.json();
  assert.equal(doc.kind, 'note');
  assert.equal(doc.asset, null);
  const updated = await (await json('PATCH', `/api/documents/${doc.id}`, { title: 'Über / lanterns', kind: 'study_guide', markdown: '# Folding\n\nKeep the corners even.' })).json();
  assert.equal(updated.kind, 'study_guide');
  const appended = await Promise.all(['Second fold.', 'Third fold.'].map((text) => json('PATCH', `/api/documents/${doc.id}`, { appendMarkdown: text })));
  assert.ok(appended.every((r) => r.status === 200));
  const stored = await (await request(`/api/documents/${doc.id}`)).json();
  assert.equal(stored.markdown, '# Folding\n\nKeep the corners even.\n\nSecond fold.\n\nThird fold.');
  assert.equal((await (await request('/api/videos/1/documents')).json()).length, 1);
  assert.deepEqual(await (await request('/api/videos/2/documents')).json(), []);
  const exported = await request(`/api/documents/${doc.id}/export`);
  assert.equal(await exported.text(), stored.markdown);
  assert.match(exported.headers.get('content-type')!, /^text\/markdown/);
  assert.match(exported.headers.get('content-disposition')!, /attachment; filename="_ber _ lanterns.md"/);
  assert.match(exported.headers.get('content-disposition')!, /filename\*=UTF-8''%C3%9Cber%20_%20lanterns.md/);
  for (const body of [{ title: ' ' }, { kind: 'attachment' }, { markdown: 'replace', appendMarkdown: 'append' }, {}]) {
    assert.equal((await json('PATCH', `/api/documents/${doc.id}`, body)).status, 400);
  }
  assert.equal((await request('/api/documents/999')).status, 404);
  assert.equal((await json('POST', '/api/videos/999/documents', { title: 'Missing' })).status, 404);
  assert.equal((await request(`/api/documents/${doc.id}`, { method: 'DELETE' })).status, 204);
  assert.equal((await request(`/api/documents/${doc.id}`)).status, 404);
});

test('uploads preserve bytes, enforce content headers, and deleting an attachment removes its asset and file', async (t) => {
  const { upload, request, json, db, dataDir } = setup(t);
  const cases: [string, Buffer, string, string][] = [
    ['paper.PDF', Buffer.from('%PDF-1.4\nInvented document bytes.\n%%EOF'), 'application/pdf', 'inline'],
    ['light.png', Buffer.from([137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3]), 'image/png', 'inline'],
    ['fold.jpg', Buffer.from([255, 216, 255, 224, 1, 2, 3]), 'image/jpeg', 'inline'],
    ['shape.webp', Buffer.from('RIFF0000WEBPtest'), 'image/webp', 'inline'],
    ['notes.docx', Buffer.from([80, 75, 3, 4, 0, 1, 2, 3]), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'attachment'],
  ];
  for (const [name, bytes, type, disposition] of cases) {
    const response = await upload(name, bytes);
    assert.equal(response.status, 201);
    const doc = await response.json();
    assert.equal(doc.kind, 'attachment');
    assert.equal(doc.markdown, null);
    assert.equal(doc.asset.mediaType, type); // The client deliberately sent text/html.
    const content = await request(`/api/assets/${doc.asset.id}/content`);
    assert.equal(content.headers.get('content-type'), type);
    assert.equal(content.headers.get('x-content-type-options'), 'nosniff');
    assert.equal(content.headers.get('content-security-policy'), 'sandbox');
    assert.ok(content.headers.get('content-disposition')!.startsWith(disposition));
    assert.equal(sha(new Uint8Array(await content.arrayBuffer())), sha(bytes));
    assert.ok((await request(`/api/assets/${doc.asset.id}/content?download=1`)).headers.get('content-disposition')!.startsWith('attachment'));
    const files = await readdir(join(dataDir, 'files'));
    assert.equal(files.length, 1);
    assert.match(files[0]!, /^[a-f0-9-]+\.[a-z]+$/);
    assert.equal(sha(await readFile(join(dataDir, 'files', files[0]!))), sha(bytes));
    assert.equal((await json('PATCH', `/api/documents/${doc.id}`, { markdown: 'Cannot edit a file' })).status, 409);
    assert.equal((await json('PATCH', `/api/documents/${doc.id}`, { kind: 'note' })).status, 409);
    assert.equal((await json('PATCH', `/api/documents/${doc.id}`, { title: 'Renamed file' })).status, 200);
    assert.equal((await request(`/api/documents/${doc.id}/export`)).status, 409);
    assert.equal((await request(`/api/documents/${doc.id}`, { method: 'DELETE' })).status, 204);
    assert.deepEqual(await readdir(join(dataDir, 'files')), []);
    assert.deepEqual(db.prepare('SELECT count(*) AS n FROM assets').get(), { n: 0 });
    assert.equal((await request(`/api/assets/${doc.asset.id}/content`)).status, 404);
  }
});

test('uploads reject disguised executable files and invalid text; Markdown and text become editable notes', async (t) => {
  const { upload, request, dataDir } = setup(t);
  for (const [name, bytes] of [['fake.pdf', '<html>Not a PDF</html>'], ['drawing.svg', '<svg/>'], ['page.html', '<html/>'], ['fake.png', '%PDF'], ['fake.docx', 'Not a ZIP']] as const) {
    const response = await upload(name, bytes);
    assert.equal(response.status, 415, name);
    assert.equal((await response.json()).error.code, 'UNSUPPORTED_FILE');
  }
  for (const bytes of [Buffer.from([255, 254, 1]), Buffer.from('binary\0text')]) assert.equal((await upload('bad.txt', bytes)).status, 415);
  for (const name of ['notes.md', 'notes.txt']) {
    const text = '# Paper lanterns\n\nÜbung: fold once.';
    const doc = await (await upload(name, text)).json();
    assert.equal(doc.kind, 'note');
    assert.equal(doc.asset, null);
    assert.equal(doc.markdown, text);
    assert.equal((await request(`/api/documents/${doc.id}/export`)).status, 200);
  }
  assert.deepEqual(await readdir(join(dataDir, 'files')), []);
  assert.equal((await upload('paper.pdf', '%PDF', 999)).status, 404);
  assert.equal((await request('/api/videos/1/assets', { method: 'POST', body: new FormData() })).status, 400);
});

test('oversize uploads are refused and a failed insert rolls back rows and cleans the managed file', async (t) => {
  const { upload, db, dataDir } = setup(t);
  assert.equal((await upload('large.pdf', new Uint8Array(25 * 1024 * 1024))).status, 413); // Multipart envelope also counts.
  db.exec("CREATE TRIGGER refuse_attachment BEFORE INSERT ON documents WHEN new.kind = 'attachment' BEGIN SELECT RAISE(ABORT, 'intentional insert failure'); END");
  const response = await upload('paper.pdf', '%PDF-1.4\nInvented bytes.');
  assert.equal(response.status, 500);
  assert.deepEqual(await readdir(join(dataDir, 'files')), []);
  assert.deepEqual(db.prepare('SELECT count(*) AS n FROM assets').get(), { n: 0 });
  assert.deepEqual(db.prepare('SELECT count(*) AS n FROM documents').get(), { n: 0 });
});

test('a failed database deletion restores the attachment file and leaves the document readable', async (t) => {
  const { upload, request, db, dataDir } = setup(t);
  const bytes = Buffer.from('%PDF-1.4\nKeep this file.');
  const doc = await (await upload('keep.pdf', bytes)).json();
  db.exec("CREATE TRIGGER refuse_delete BEFORE DELETE ON documents BEGIN SELECT RAISE(ABORT, 'intentional delete failure'); END");
  assert.equal((await request(`/api/documents/${doc.id}`, { method: 'DELETE' })).status, 500);
  assert.equal((await request(`/api/documents/${doc.id}`)).status, 200);
  const content = await request(`/api/assets/${doc.asset.id}/content`);
  assert.equal(sha(new Uint8Array(await content.arrayBuffer())), sha(bytes));
  assert.ok((await readdir(join(dataDir, 'files'))).every((name) => !name.startsWith('tmp-')));
});
