import { Link } from 'react-router';
import { formatTime } from '../../../../shared/time.ts';
import { useJob, type VideoSummary } from '../../api.ts';
import { transcriptStatus } from './status.ts';

function VideoCard({ video, onChanged }: { video: VideoSummary; onChanged: () => void }) {
  const job = useJob(video.activeJob?.id, onChanged);
  const status = transcriptStatus(video, job);
  const href = `/videos/${video.id}`;
  return (
    <article className="video-card">
      <Link to={href} className="thumb" tabIndex={-1} aria-hidden="true">
        {video.thumbnailUrl && <img src={video.thumbnailUrl} alt="" loading="lazy" />}
        {video.durationSeconds !== null && <span className="duration">{formatTime(video.durationSeconds)}</span>}
      </Link>
      <Link to={href} className="title">
        {video.title}
      </Link>
      <div className="meta">
        {video.channel && <span className="muted">{video.channel}</span>}
        <span className={`status ${status.tone === 'ok' ? 'ok' : status.tone === 'problem' ? 'problem' : ''}`} role="status">
          {status.text}
        </span>
      </div>
      {video.topics.length > 0 && (
        <div className="meta">
          {video.topics.map((topic) => (
            <Link key={topic.id} to={`/topics/${topic.id}`} className="chip">
              {topic.name}
            </Link>
          ))}
        </div>
      )}
    </article>
  );
}

export function VideoGrid({ videos, onChanged }: { videos: VideoSummary[]; onChanged: () => void }) {
  return (
    <div className="video-grid">
      {videos.map((video) => (
        <VideoCard key={video.id} video={video} onChanged={onChanged} />
      ))}
    </div>
  );
}
