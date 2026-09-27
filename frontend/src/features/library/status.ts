import type { Job, VideoSummary } from '../../api.ts';

export type StatusLabel = { text: string; tone: 'ok' | 'busy' | 'problem' | 'neutral' };

/** Human wording for a video's transcript state, preferring a live job's stage. */
export function transcriptStatus(video: VideoSummary, job?: Pick<Job, 'status' | 'stage'> | null): StatusLabel {
  const active = job && (job.status === 'queued' || job.status === 'running') ? job : video.activeJob;
  if (active && active.status === 'queued') return { text: 'Waiting to fetch transcript…', tone: 'busy' };
  if (active && active.status === 'running') {
    return { text: active.stage === 'saving' ? 'Saving transcript…' : 'Fetching transcript…', tone: 'busy' };
  }
  switch (video.transcriptStatus) {
    case 'ready':
      return { text: 'Ready', tone: 'ok' };
    case 'no_captions':
      return { text: 'No captions', tone: 'problem' };
    case 'blocked':
      return { text: 'Blocked by YouTube', tone: 'problem' };
    case 'failed':
      return { text: 'Failed', tone: 'problem' };
    default:
      return { text: 'Transcript not fetched', tone: 'neutral' };
  }
}
