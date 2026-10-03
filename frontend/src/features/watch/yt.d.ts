declare namespace YT {
  type PlayerEvent = { target: Player };
  type StateEvent = PlayerEvent & { data: number };
  type Options = {
    videoId: string;
    width?: string;
    height?: string;
    playerVars: { start: number; rel: number; origin: string };
    events: { onReady(event: PlayerEvent): void; onStateChange(event: StateEvent): void; onError(event: StateEvent): void };
  };
  class Player {
    constructor(element: HTMLElement, options: Options);
    getCurrentTime(): number;
    getDuration(): number;
    getPlayerState(): number;
    getIframe(): HTMLIFrameElement;
    seekTo(seconds: number, allowSeekAhead: boolean): void;
    destroy(): void;
  }
}

interface Window {
  YT?: { Player: typeof YT.Player };
  onYouTubeIframeAPIReady?: () => void;
}
