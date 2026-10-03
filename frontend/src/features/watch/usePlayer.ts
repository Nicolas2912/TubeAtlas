import { useCallback, useEffect, useRef, useState } from 'react';
import { api, errorMessage, unwrap, type VideoSummary } from '../../api.ts';
import { clampPlayback } from '../../../../shared/watch.ts';
import { loadPlayerApi } from './player.ts';

export function usePlayer(video: VideoSummary, initialTime: number) {
  const container = useRef<HTMLDivElement>(null);
  const player = useRef<YT.Player | null>(null);
  const position = useRef(initialTime);
  const duration = useRef(video.durationSeconds);
  duration.current = video.durationSeconds;
  const [time, setTime] = useState(initialTime);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  const savePosition = useRef<(seconds: number) => void>(() => {});

  const seek = useCallback((seconds: number) => {
    if (!player.current) return;
    const target = clampPlayback(seconds, duration.current ?? (player.current.getDuration() || null));
    player.current.seekTo(target, true);
    position.current = target;
    setTime(target);
    savePosition.current(target);
  }, []);

  useEffect(() => {
    const host = container.current!;
    let alive = true;
    let instance: YT.Player | undefined;
    let tick: ReturnType<typeof setInterval> | undefined;
    let readyTimeout: ReturnType<typeof setTimeout> | undefined;
    let failed = false;
    let played = false;
    let lastSave = performance.now();
    let queue = Promise.resolve();
    setReady(false);
    setError(null);

    const save = (seconds: number) => {
      queue = queue.then(async () => {
        try {
          await unwrap(api.videos[':id'].$patch({ param: { id: String(video.id) }, json: { playbackSeconds: seconds } }, { init: { keepalive: true } }));
          if (alive) setSaveError(null);
        } catch (err) { if (alive) setSaveError(errorMessage(err)); }
      });
    };
    savePosition.current = save;
    const read = () => {
      const seconds = instance?.getCurrentTime();
      if (seconds === undefined || !Number.isFinite(seconds)) return;
      position.current = clampPlayback(seconds, duration.current ?? (instance?.getDuration() || null));
      if (alive) setTime(position.current);
    };
    const saveOnLeave = () => { if (played) { read(); save(position.current); } };
    window.addEventListener('pagehide', saveOnLeave);

    void loadPlayerApi().then(() => {
      if (!alive) return;
      const element = document.createElement('div');
      host.replaceChildren(element);
      readyTimeout = setTimeout(() => {
        if (!alive) return;
        failed = true;
        setError('YouTube did not respond. You can keep reading the transcript.');
      }, 15000);
      instance = new window.YT!.Player(element, {
        videoId: video.youtubeId, width: '100%', height: '100%',
        playerVars: { start: Math.floor(position.current), rel: 0, origin: window.location.origin },
        events: {
          onReady: ({ target }) => {
            if (!alive || failed) return;
            clearTimeout(readyTimeout);
            player.current = target;
            target.getIframe().title = `YouTube player: ${video.title}`;
            setReady(true);
            tick = setInterval(() => {
              if (target.getPlayerState() !== 1) return;
              read();
              if (performance.now() - lastSave >= 10000) { lastSave = performance.now(); save(position.current); }
            }, 250);
          },
          onStateChange: ({ data }) => {
            if (!alive || failed) return;
            if (data === 1) { played = true; read(); }
            if ((data === 2 || data === 0) && played) { read(); save(position.current); lastSave = performance.now(); }
          },
          onError: ({ data }) => {
            if (!alive) return;
            failed = true;
            clearTimeout(readyTimeout);
            clearInterval(tick);
            setReady(false);
            player.current = null;
            setError(data === 101 || data === 150 ? "This video can't be played here." : data === 100 ? 'This video is unavailable on YouTube.' : 'YouTube could not play this video.');
          },
        },
      });
    }).catch((err: unknown) => { if (alive) setError(errorMessage(err)); });

    return () => {
      saveOnLeave();
      alive = false;
      clearInterval(tick);
      clearTimeout(readyTimeout);
      window.removeEventListener('pagehide', saveOnLeave);
      player.current = null;
      instance?.destroy();
      host.replaceChildren();
    };
  }, [video.id, video.youtubeId, version]);

  return { container, time, ready, error, saveError, seek, retry: () => setVersion((v) => v + 1), retrySave: () => savePosition.current(position.current) };
}
