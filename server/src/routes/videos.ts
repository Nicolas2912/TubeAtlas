import { Hono } from 'hono';
import { ImportVideoBody, IdParam, UpdateVideoBody, VideoListQuery } from '../../../shared/api.ts';
import type { ChatService } from '../services/chat.ts';
import type { AppDeps } from '../app.ts';
import { importVideo } from '../services/import.ts';
import { deleteVideo, getVideo, listVideos, updateVideo } from '../services/videos.ts';
import { validate } from '../validate.ts';

export function videoRoutes({ db, config, youtube, jobs, chat }: AppDeps & { chat: ChatService }) {
  return new Hono()
    .get('/', validate('query', VideoListQuery), (c) => c.json(listVideos(db, c.req.valid('query').topicId)))
    .post('/import', validate('json', ImportVideoBody), async (c) => {
      const { created, video, job } = await importVideo({ db, youtube, jobs }, c.req.valid('json'));
      return c.json({ video, job }, created ? 201 : 200);
    })
    .get('/:id', validate('param', IdParam), (c) => c.json(getVideo(db, c.req.valid('param').id)))
    .patch('/:id', validate('param', IdParam), validate('json', UpdateVideoBody), (c) =>
      c.json(updateVideo(db, c.req.valid('param').id, c.req.valid('json'))),
    )
    .delete('/:id', validate('param', IdParam), (c) => {
      const { id } = c.req.valid('param');
      jobs.cancelForVideo(id);
      chat.cancelForVideo(id);
      deleteVideo(db, config.dataDir, id);
      return c.body(null, 204);
    });
}
