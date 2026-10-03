import { Link, NavLink, Outlet, useOutletContext, useParams } from 'react-router';
import { api, unwrap, useApi, useJob, type Job, type VideoSummary } from '../api.ts';
import { TopicPicker } from '../components/TopicPicker.tsx';
import { LoadError } from '../components/LoadError.tsx';
import { formatTime } from '../../../shared/time.ts';

export type VideoContext = { video: VideoSummary; job: Job | null; reload: () => void };

const tabs = [{ label: 'Watch & Read', to: 'watch' }, { label: 'Documents', to: 'documents' }];

export function useVideoContext() { return useOutletContext<VideoContext>(); }

export default function VideoLayout() {
  const { videoId = '' } = useParams();
  const video = useApi(() => unwrap(api.videos[':id'].$get({ param: { id: videoId } })), [videoId]);
  const job = useJob(video.data?.activeJob?.id, video.reload);
  if (!video.data) return video.error ? <LoadError error={video.error} retry={video.reload} /> : <p role="status">Loading video…</p>;
  const data = video.data;
  const topic = data.topics[0];
  return (
    <>
      <nav className="breadcrumb" aria-label="Breadcrumb">
        <Link to={topic ? `/topics/${topic.id}` : '/'}>{topic?.name ?? 'Library'}</Link><span aria-hidden="true">/</span><span>{data.title}</span>
      </nav>
      <div className="video-header">
        <div>
          <div className="video-title"><h1>{data.title}</h1>{data.durationSeconds !== null && <span className="muted">· {formatTime(data.durationSeconds)}</span>}</div>
          <div className="video-subline"><span className="muted">{data.channel}</span>{data.topics.map((t) => <Link key={t.id} to={`/topics/${t.id}`} className="chip">{t.name}</Link>)}</div>
        </div>
        <div className="video-actions"><TopicPicker key={data.id} video={data} onChanged={video.reload} /></div>
      </div>
      {tabs.length > 0 && <nav className="tabs" aria-label="Video views">
        {tabs.map((tab) => <NavLink key={tab.to} to={tab.to} className="tab">{tab.label}</NavLink>)}
      </nav>}
      {video.error !== undefined && <LoadError error={video.error} retry={video.reload} />}
      <Outlet context={{ video: data, job, reload: video.reload } satisfies VideoContext} />
    </>
  );
}
