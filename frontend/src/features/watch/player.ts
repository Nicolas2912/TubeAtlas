let loading: Promise<void> | undefined;

/** The shared API script is loaded only when the reader is opened. Failed loads can be retried. */
export function loadPlayerApi(): Promise<void> {
  if (window.YT?.Player) return Promise.resolve();
  if (loading) return loading;
  loading = new Promise<void>((resolve, reject) => {
    const script = document.createElement('script');
    const previous = window.onYouTubeIframeAPIReady;
    const finish = (error?: Error) => {
      clearTimeout(timeout);
      window.onYouTubeIframeAPIReady = previous;
      script.onerror = null;
      if (error) { script.remove(); reject(error); }
      else resolve();
    };
    window.onYouTubeIframeAPIReady = () => { finish(); previous?.(); };
    const timeout = window.setTimeout(() => finish(new Error('YouTube did not respond.')), 15000);
    script.src = 'https://www.youtube.com/iframe_api';
    script.onerror = () => finish(new Error('Could not load the YouTube player.'));
    document.head.append(script);
  }).catch((error: unknown) => { loading = undefined; throw error; });
  return loading;
}
