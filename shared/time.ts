/** Formats media seconds as m:ss, or h:mm:ss from one hour on. */
export function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

/** App route that opens a video's Watch & Read view at a moment. */
export function watchLink(videoId: number, seconds: number): string {
  return `/videos/${videoId}/watch?t=${Math.max(0, Math.floor(seconds))}`;
}
