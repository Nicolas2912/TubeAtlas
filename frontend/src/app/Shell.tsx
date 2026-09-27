import { createContext, useContext, useEffect, useRef, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router';
import { Icon } from '../components/Icon.tsx';
import { ImportDialog } from '../features/library/ImportDialog.tsx';

const ImportContext = createContext<() => void>(() => {});
/** Opens the global "Import video" dialog. */
export const useOpenImport = () => useContext(ImportContext);

export function Shell() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const location = useLocation();
  const menuButton = useRef<HTMLButtonElement>(null);

  useEffect(() => setMenuOpen(false), [location.pathname]);
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setMenuOpen(false);
        menuButton.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [menuOpen]);

  const navClass = ({ isActive }: { isActive: boolean }) => `nav-link${isActive ? ' active' : ''}`;

  return (
    <ImportContext.Provider value={() => setImportOpen(true)}>
      <div className={`shell${menuOpen ? ' menu-open' : ''}`}>
        <nav className="sidebar" aria-label="Main">
          <Link to="/" className="logo">
            <Icon name="logo" size={22} /> TubeAtlas
          </Link>
          <NavLink to="/" end className={navClass}>
            <Icon name="library" /> Library
          </NavLink>
          <NavLink to="/topics" className={navClass}>
            <Icon name="topic" /> Topics
          </NavLink>
          <div className="spacer" />
          <NavLink to="/settings" className={navClass}>
            <Icon name="settings" /> Settings
          </NavLink>
        </nav>
        <div className="backdrop" onClick={() => setMenuOpen(false)} />
        <div className="main">
          <header className="topbar">
            <button ref={menuButton} className="button menu-button" aria-expanded={menuOpen} onClick={() => setMenuOpen((o) => !o)}>
              <Icon name="menu" /> <span className="visually-hidden">Menu</span>
            </button>
            <button className="button primary" onClick={() => setImportOpen(true)}>
              <Icon name="plus" /> Import video
            </button>
          </header>
          <main className="content">
            <Outlet />
          </main>
        </div>
        <ImportDialog open={importOpen} onClose={() => setImportOpen(false)} />
      </div>
    </ImportContext.Provider>
  );
}
