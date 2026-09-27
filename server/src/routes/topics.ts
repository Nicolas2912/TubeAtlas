import { Hono } from 'hono';
import { IdParam, TopicBody } from '../../../shared/api.ts';
import type { AppDeps } from '../app.ts';
import { createTopic, deleteTopic, listTopics, renameTopic } from '../services/topics.ts';
import { validate } from '../validate.ts';

export function topicRoutes({ db }: AppDeps) {
  return new Hono()
    .get('/', (c) => c.json(listTopics(db)))
    .post('/', validate('json', TopicBody), (c) => c.json(createTopic(db, c.req.valid('json').name), 201))
    .patch('/:id', validate('param', IdParam), validate('json', TopicBody), (c) =>
      c.json(renameTopic(db, c.req.valid('param').id, c.req.valid('json').name)),
    )
    .delete('/:id', validate('param', IdParam), (c) => {
      deleteTopic(db, c.req.valid('param').id);
      return c.body(null, 204);
    });
}
