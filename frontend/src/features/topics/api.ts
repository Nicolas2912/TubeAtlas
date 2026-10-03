import { api, ApiError, unwrap, type Topic } from '../../api.ts';

/** Finds or creates a topic by name (case-insensitive, like the server). */
export async function ensureTopic(name: string): Promise<Topic> {
  try {
    return await unwrap(api.topics.$post({ json: { name } }));
  } catch (err) {
    if (!(err instanceof ApiError) || err.code !== 'TOPIC_EXISTS') throw err;
    const topics = await unwrap(api.topics.$get());
    const existing = topics.find((t) => t.name.toLowerCase() === name.trim().toLowerCase());
    if (!existing) throw err;
    return existing;
  }
}
