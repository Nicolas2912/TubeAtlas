import type { Db } from '../db.ts';
import { AppError } from '../errors.ts';

export type Topic = { id: number; name: string; videoCount: number };

export function listTopics(db: Db): Topic[] {
  return db
    .prepare(
      `SELECT t.id, t.name, count(vt.video_id) AS videoCount
       FROM topics t LEFT JOIN video_topics vt ON vt.topic_id = t.id
       GROUP BY t.id ORDER BY t.name COLLATE NOCASE`,
    )
    .all() as Topic[];
}

function getTopic(db: Db, id: number): Topic {
  const topic = listTopics(db).find((t) => t.id === id);
  if (!topic) throw new AppError(404, 'NOT_FOUND', 'Topic not found.');
  return topic;
}

function uniqueName<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    if (err instanceof Error && /UNIQUE constraint failed: topics\.name/.test(err.message)) {
      throw new AppError(409, 'TOPIC_EXISTS', 'A topic with that name already exists.');
    }
    throw err;
  }
}

export function createTopic(db: Db, name: string): Topic {
  const id = uniqueName(() => Number(db.prepare('INSERT INTO topics (name) VALUES (?)').run(name).lastInsertRowid));
  return getTopic(db, id);
}

export function renameTopic(db: Db, id: number, name: string): Topic {
  getTopic(db, id);
  uniqueName(() => db.prepare('UPDATE topics SET name = ? WHERE id = ?').run(name, id));
  return getTopic(db, id);
}

/** Deletes the topic; its videos stay in the library. */
export function deleteTopic(db: Db, id: number) {
  getTopic(db, id);
  db.prepare('DELETE FROM topics WHERE id = ?').run(id);
}
