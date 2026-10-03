import { useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router';
import { useVideoContext, type VideoContext } from '../../app/VideoLayout.tsx';
import { api, ApiError, errorMessage, unwrap, useApi } from '../../api.ts';
import { LoadError } from '../../components/LoadError.tsx';
import { transcriptStatus } from '../library/status.ts';
import { playbackTarget } from '../../../../shared/watch.ts';
import { formatTime } from '../../../../shared/time.ts';
import { usePlayer } from './usePlayer.ts';
import { TranscriptPanel } from './TranscriptPanel.tsx';
import { ManualTranscript } from './ManualTranscript.tsx';

function WatchReader({ video, job, reload }: VideoContext) {
  const [params] = useSearchParams();
  const requested = params.get('t');
  const previousRequest = useRef(requested);
  const initialTime = useRef(playbackTarget(requested, video.playbackSeconds, video.durationSeconds));
  const player = usePlayer(video, initialTime.current);
  const transcript = useApi(async () => {
    try { return await unwrap(api.videos[':id'].transcript.$get({ param: { id: String(video.id) } })); }
    catch (err) { if (err instanceof ApiError && err.code === 'NO_TRANSCRIPT') return null; throw err; }
  }, [video.id, video.transcriptStatus]);
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);
  const status = transcriptStatus(video, job);

  useEffect(() => {
    if (!player.ready || requested === previousRequest.current) return;
    previousRequest.current = requested;
    if (requested !== null) player.seek(playbackTarget(requested, player.time, video.durationSeconds));
  }, [requested, player.ready, player.seek, video.durationSeconds, player.time]);

  async function retryTranscript() {
    setRetrying(true);
    setRetryError(null);
    try { await unwrap(api.videos[':id'].transcript.retry.$post({ param: { id: String(video.id) } })); reload(); }
    catch (err) { setRetryError(errorMessage(err)); }
    finally { setRetrying(false); }
  }

  return <div className="watch-layout">
    <section className="watch-player" aria-label="Video player">
      <div className="player-frame"><div ref={player.container} className="youtube-player" />
        {!player.ready && !player.error && <p className="player-loading" role="status">Loading YouTube player…</p>}
      </div>
      {player.error && <div className="notice error-box" role="alert"><p>{player.error}</p><button className="button small" onClick={player.retry}>Try player again</button></div>}
      <div className="playback-detail"><span className="muted">Position {formatTime(player.time)}{video.durationSeconds !== null ? ` / ${formatTime(video.durationSeconds)}` : ''}</span>
        <a className="link-button" href={`https://www.youtube.com/watch?v=${video.youtubeId}&t=${Math.floor(player.time)}`} target="_blank" rel="noreferrer">Open on YouTube</a>
      </div>
      {player.saveError && <p className="error" role="alert">Playback position could not be saved. {player.saveError} <button className="link-button" onClick={player.retrySave}>Try saving again</button></p>}
      <p className="muted reader-hint">Click a transcript timestamp to jump to that moment. Your playback position is saved as you watch.</p>
    </section>
    {transcript.error !== undefined ? <LoadError error={transcript.error} retry={transcript.reload} /> : transcript.data ?
      <TranscriptPanel key={transcript.data.transcriptId} transcript={transcript.data} video={video} time={player.time} ready={player.ready} seek={player.seek} /> :
      <section className="panel transcript-fallback" aria-label="Transcript">
        <h2>Transcript</h2>
        {transcript.loading ? <p role="status">Loading transcript…</p> : <>
          <p className={`status ${status.tone === 'problem' ? 'problem' : ''}`} role="status">{status.text}</p>
          {video.transcriptError && <p className="muted">{video.transcriptError}</p>}
          {(video.transcriptStatus === 'blocked' || video.transcriptStatus === 'failed') && <button className="button" disabled={retrying || video.activeJob !== null} onClick={() => void retryTranscript()}>{retrying ? 'Requesting captions…' : 'Try again'}</button>}
          {retryError && <p className="error" role="alert">{retryError}</p>}
          <ManualTranscript videoId={video.id} onSaved={() => { transcript.reload(); reload(); }} />
        </>}
      </section>}
  </div>;
}

export default function WatchPage() {
  const context = useVideoContext();
  return <WatchReader key={context.video.id} {...context} />;
}
