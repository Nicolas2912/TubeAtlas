import { createContext, Suspense, useContext, useEffect, useRef, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router';
import { Icon } from '../components/Icon.tsx';
import { ImportDialog } from '../features/library/ImportDialog.tsx';
import { trapDialogFocus } from '../components/dialog.ts';

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
  const [menuOpen, setMenuOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const menu = useRef<HTMLDialogElement>(null);
  const location = useLocation();

  useEffect(() => { menu.current?.close(); }, [location.pathname]);
  useEffect(() => {
    const wide = window.matchMedia('(min-width: 901px)');
    const close = () => { if (wide.matches) menu.current?.close(); };
    wide.addEventListener('change', close);
    return () => wide.removeEventListener('change', close);
  }, []);

  return (
    <ImportContext.Provider value={() => setImportOpen(true)}>
      <a href="#content" className="skip-link">Skip to content</a>
      <div className="shell">
        <aside className="sidebar"><Navigation /></aside>
        <dialog ref={menu} id="mobile-menu" className="mobile-menu" aria-label="Main navigation" onKeyDown={trapDialogFocus} onClose={() => setMenuOpen(false)}>
          <form method="dialog"><button className="button small">Close menu</button></form>
          <Navigation />
        </dialog>
        <div className="main">
          <header className="topbar">
            <button className="button menu-button" aria-controls="mobile-menu" aria-expanded={menuOpen} onClick={() => { menu.current?.showModal(); setMenuOpen(true); }}>
              <Icon name="menu" /> <span className="visually-hidden">Menu</span>
            </button>
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
