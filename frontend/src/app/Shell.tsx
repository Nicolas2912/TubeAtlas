import { createContext, Suspense, useContext, useState } from 'react';
import { Link, NavLink, Outlet } from 'react-router';
import { Icon } from '../components/Icon.tsx';
import { ImportDialog } from '../features/library/ImportDialog.tsx';

const ImportContext = createContext<() => void>(() => {});
export const useOpenImport = () => useContext(ImportContext);

function Navigation() {
  const navClass = ({ isActive }: { isActive: boolean }) => `nav-link${isActive ? ' active' : ''}`;
  return (
    <nav className="navigation" aria-label="Main">
      <Link to="/" className="logo"><Icon name="logo" size={22} /> TubeAtlas</Link>
      <NavLink to="/" end className={navClass}><Icon name="library" /> Library</NavLink>
      <NavLink to="/topics" className={navClass}><Icon name="topic" /> Topics</NavLink>
      <div className="spacer" />
      <NavLink to="/settings" className={navClass}><Icon name="settings" /> Settings</NavLink>
    </nav>
  );
}

export function Shell() {
  const [importOpen, setImportOpen] = useState(false);

  return (
    <ImportContext.Provider value={() => setImportOpen(true)}>
      <a href="#content" className="skip-link">Skip to content</a>
      <div className="shell">
        <aside className="sidebar"><Navigation /></aside>
        <div className="main">
          <header className="topbar">
            <input className="input global-search" type="search" placeholder="Search your knowledge…" aria-label="Search your knowledge (coming later)" disabled />
            <button className="button primary" onClick={() => setImportOpen(true)}><Icon name="plus" /> Import video</button>
          </header>
          <main id="content" className="content" tabIndex={-1}>
            <Suspense fallback={<p role="status">Loading…</p>}><Outlet /></Suspense>
          </main>
        </div>
        <ImportDialog open={importOpen} onClose={() => setImportOpen(false)} />
      </div>
    </ImportContext.Provider>
  );
}
