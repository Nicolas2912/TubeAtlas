import { useOpenImport } from '../../app/Shell.tsx';
import { api, unwrap, useApi } from '../../api.ts';
import { Icon } from '../../components/Icon.tsx';
import { VideoGrid } from './VideoGrid.tsx';
import { LoadError } from '../../components/LoadError.tsx';

export default function LibraryPage({ topicId, title = 'Library' }: { topicId?: string; title?: string }) {
  const openImport = useOpenImport();
  const videos = useApi(() => unwrap(api.videos.$get({ query: topicId ? { topicId } : {} })), [topicId]);

  return (
    <>
      <div className="page-header">
        <h1>{title}</h1>
        {videos.data && videos.data.length > 0 && <span className="muted">{videos.data.length} {videos.data.length === 1 ? 'video' : 'videos'}</span>}
      </div>
      {videos.error !== undefined && <LoadError error={videos.error} retry={videos.reload} />}
      {videos.data?.length === 0 && (
        <div className="empty">
          <h2>{topicId ? 'No videos in this topic' : 'Import your first video'}</h2>
          <p className="muted">{topicId ? 'Import a video here, or add this topic to a video in your library.' : 'Paste a YouTube link to start your library.'}</p>
          <button className="button primary" onClick={openImport}>
            <Icon name="plus" /> Import video
          </button>
        </div>
      )}
      {videos.data && videos.data.length > 0 && <VideoGrid videos={videos.data} onChanged={videos.reload} />}
      {videos.loading && !videos.data && <p className="muted" role="status">Loading videos…</p>}
    </>
  );
}
