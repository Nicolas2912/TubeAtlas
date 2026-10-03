import { errorMessage } from '../api.ts';

export function LoadError({ error, retry }: { error: unknown; retry: () => void }) {
  return (
    <div className="notice error-box" role="alert">
      {errorMessage(error)}{' '}<button className="link-button" onClick={retry}>Try again</button>
    </div>
  );
}
