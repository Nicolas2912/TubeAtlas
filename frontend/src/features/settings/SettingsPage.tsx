import { api, unwrap, useApi } from '../../api.ts';
import { LoadError } from '../../components/LoadError.tsx';

export default function SettingsPage() {
  const settings = useApi(() => unwrap(api.settings.$get()), []);
  return (
    <>
      <div className="page-header"><h1>Settings</h1></div>
      {settings.error !== undefined && <LoadError error={settings.error} retry={settings.reload} />}
      {settings.loading && !settings.data && <p role="status">Loading settings…</p>}
      {settings.data && <section className="panel settings-panel" aria-label="Configuration">
        <h2>Configuration</h2>
        <dl className="settings">
          <dt>OpenRouter key</dt><dd>{settings.data.aiConfigured ? 'Configured' : 'Not configured'}</dd>
          <dt>YouTube API key</dt><dd>{settings.data.youtubeKey ? 'Configured' : 'Not configured (imports still work)'}</dd>
          <dt>Chat model</dt><dd><code>{settings.data.models.chat}</code></dd>
          <dt>Embedding model</dt><dd><code>{settings.data.models.embedding}</code></dd>
          <dt>Knowledge graph model</dt><dd><code>{settings.data.models.kg}</code></dd>
          <dt>Reasoning effort</dt><dd>{settings.data.models.kgReasoningEffort}</dd>
          <dt>Data directory</dt><dd><code>{settings.data.dataDir}</code></dd>
        </dl>
      </section>}
      <section className="panel settings-panel">
        <h2>Back up your library</h2>
        <p>Stop TubeAtlas, then copy the entire data directory shown above. It contains your database and files. To restore it, stop the app and replace that directory with your backup.</p>
        <p className="muted">Keys and models are configured in your local .env file. Restart the app after changing them.</p>
      </section>
    </>
  );
}
