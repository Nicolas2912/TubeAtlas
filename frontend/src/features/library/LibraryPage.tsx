import { useOpenImport } from '../../app/Shell.tsx';
import { api, errorMessage, unwrap, useApi } from '../../api.ts';
import { Icon } from '../../components/Icon.tsx';
import { VideoGrid } from './VideoGrid.tsx';

export default function LibraryPage() {
  const openImport = useOpenImport();
  const videos = useApi(() => unwrap(api.videos.$get({ query: {} })), []);

  return (
    <>
      <div className="page-header">
        <h1>Library</h1>
        {videos.data && videos.data.length > 0 && <span className="muted">{videos.data.length} videos</span>}
      </div>
      {videos.error !== undefined && !videos.data && (
        <div className="notice error-box" role="alert">
          {errorMessage(videos.error)}{' '}
          <button className="link-button" onClick={videos.reload}>
            Try again
          </button>
        </div>
      )}
      {videos.data?.length === 0 && (
        <div className="empty">
          <h2>Import your first video</h2>
          <p className="muted">Paste a YouTube link to watch it beside its transcript and keep notes.</p>
          <button className="button primary" onClick={openImport}>
            <Icon name="plus" /> Import video
          </button>
        </div>
      )}
      {videos.data && videos.data.length > 0 && <VideoGrid videos={videos.data} onChanged={videos.reload} />}
      {videos.loading && !videos.data && <p className="muted">Loading…</p>}
    </>
  );
}
