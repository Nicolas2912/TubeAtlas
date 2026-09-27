# Work log

One section per work package from [work-packages.md](work-packages.md): what was done, how it was verified, and any deviations.

## WP-00: Cutover and Node skeleton (2026-09-27)

**Done**
- Removed the Python application and tooling: `src/`, `tests/`, `scripts/check_api_connections.py`, Poetry/pytest/mypy/flake8 config, pre-commit and detect-secrets config, both Dockerfiles, both Compose files, the Python CI workflows, `.github/README.md` (a copy of the Python README), Sphinx docs (`docs/source/`, `docs/Makefile`), `.env.template`, and the old sample data `data/raw/`. Deleted the empty local `tubeatlas.db` and the untracked `.venv/` and `.pytest_cache/`.
- Added the Node 26 project: `package.json` (scripts per §3.3, `engines >=26.10`), `.nvmrc`, `tsconfig.base.json`, `server/tsconfig.json`, `frontend/tsconfig.json`, `.env.example`, a Node-only `.gitignore`, `.github/workflows/ci.yml`, and a new `README.md`.
- Dependencies exactly per §3.2; `youtube-transcript` pinned to `1.3.1`.
- Entry points: `server/src/app.ts` (`/api/health`, JSON 404 for unknown API paths, static frontend with SPA fallback), `server/src/main.ts` (binds 127.0.0.1:5170), `frontend/` (Vite + React page "TubeAtlas", `/api` proxy), `scripts/dev.ts`, and `shared/time.ts` with tests.
- The design references `output/ui-concepts/` were already committed (commit `eeb0f10`).
- Local Node upgraded to 26.10.0 via mise (global setting `node = "26"`).

**Verification**
- `npm run check`: both typechecks pass; Vite build OK.
- `npm test`: 4/4 pass (`formatTime`/`watchLink`; health; API 404).
- `npm start` after `npm run build`: `/api/health` → `{"ok":true}`; `/` and `/topics` → the built page (200); built JS asset → 200 `text/javascript`; `/api/nope` → 404 JSON `NOT_FOUND`; listening socket is `127.0.0.1:5170` only.
- `npm run dev`: http://127.0.0.1:5173/ serves the page; `/api/health` through the Vite proxy → `{"ok":true}`; SIGINT to the launcher stopped the API watcher, the API server, and Vite.
- Fresh clone of the WP-00 commit (`git clone` to /tmp): `npm ci` (121 packages, no install-script warnings) → `npm run check` passes → `npm test` 4/4 pass.

**Deviations and notes**
- The criterion "`git ls-files` contains no `.py` files" can only hold **outside `legacy/`**, because `legacy/` must stay unchanged and contains Python files. Checked that way.
- Found and fixed during verification: unknown `/api/...` paths returned the SPA page with 200, because a mounted Hono sub-app's `notFound` handler isn't used. An explicit `app.all('/api/*')` 404 now precedes the static routes (covered by a test).
- npm 11 on Node 26 asks for approval of install scripts. `better-sqlite3@13` ships prebuilt binaries for linux/darwin/win32 (x64 and arm64), so its `node-gyp rebuild` script is **denied** (`allowScripts: { "better-sqlite3": false }`); the prebuilt binary loads. This avoids needing a compiler, including in CI.
