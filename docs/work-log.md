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

## WP-01: Database, config, server shell (2026-09-27)

**Done**
- `server/src/config.ts`: Zod-validated, frozen `Config` per §3.4. Empty values count as unset, `GOOGLE_API_KEY` is an alias for `YOUTUBE_API_KEY`, and `DATA_DIR` is resolved to an absolute path. Errors name variables, never values.
- `server/src/db.ts`: `open()` (creates `DATA_DIR` and `files/`; WAL, foreign keys, 5 s busy timeout, `synchronous=NORMAL`), `migrate()` (ordered `NNN_name.sql`, each in its own transaction, SHA-256 recorded; `MIGRATION_EDITED`, `MIGRATION_MISSING`, bad-name and duplicate-version errors), `tx()`, `recoverInterrupted()`, and the `--migrate` entry point (`npm run db:migrate`).
- `server/migrations/001_init.sql`: extracted verbatim from the SQL block in work-packages.md, so the schema matches the spec exactly (11 tables plus indexes).
- `server/src/errors.ts`: `AppError`, `errorBody()`, `handleError()` (unknown errors become a generic `500 INTERNAL`, details logged server-side only).
- `server/src/app.ts`: `createApp({ db, config })`. Host guard (`403 BAD_HOST`) and Origin guard for state-changing requests (`403 BAD_ORIGIN`) on everything, a 1 MB body limit on `/api` (`413 PAYLOAD_TOO_LARGE`), no CORS, and `/api/health` → `{ ok, aiConfigured, youtubeKey }`. The JSON 404 moved from WP-00's `app.all('/api/*')` to the root `notFound`, with the frontend handlers skipping `/api` paths.
- `server/src/main.ts`: load config → open → migrate → recover → serve on 127.0.0.1; closes the server and database on SIGINT/SIGTERM.
- `server/src/testing.ts`: `createTestApp()` for server tests.
- Dev web port moved from 5173 to **5171** (`shared/ports.ts`, used by both `vite.config.ts` and the host guard).

**Verification**
- `npm run check` passes; `npm test` 19/19 pass:
  - config: defaults, alias, errors without values;
  - db: open and pragmas; fresh migrate plus no-op rerun; edited and missing migrations; a broken migration rolls back while earlier ones stay applied; cascade from a video to all 9 dependent tables; the three partial unique indexes; the documents `CHECK`s; recovery;
  - app: health without secrets; JSON 404 for GET/POST; foreign Host (also a foreign port) → 403; foreign Origin POST → 403, while same-origin, no-Origin and cross-origin GET pass; `AppError` code kept; unknown error → 500 without its message; > 1 MB body → 413.
- Real server on a fresh `DATA_DIR`: health OK; `curl -H 'Host: evil.example'` → 403 `BAD_HOST`; POST with `Origin: https://evil.example` → 403 `BAD_ORIGIN`; `/api/nope` → JSON 404; `/videos/1/watch` → 200 page; database created with `001_init.sql` applied (12 tables including `schema_migrations`).
- Restart recovery across real processes: inserted a `running` job, restarted → log "Marked interrupted after restart: 1 job(s), 0 message(s)"; the job is `interrupted` with `finished_at` set; SIGINT exits with code 0.
- `npm run db:migrate` on the real `data/` applied `001_init.sql`; a second run printed "Database is up to date."
- Deleted only `data/tubeatlas.sqlite*` and ran `npm run dev`: the database was recreated and migrated (WAL); `data/evaluation/` untouched. Through the Vite proxy on 5171: page OK, `/api/health` 200, same-origin POST reaches the API (JSON 404).

**Deviations and notes**
- The criterion "Deleting `data/` and starting again" was changed to deleting only the database files or using an empty `DATA_DIR`. Deleting `data/` would destroy the evaluation transcript snapshots in `data/evaluation/`.
- Startup recovery lives in `recoverInterrupted(db)`, called from `main.ts`, rather than inside `createApp`. That keeps app creation free of side effects; the criterion wording was updated.
- The host guard always allows the dev web port, not only in dev (no security cost; avoids a mode switch).
- Port 5173 was in use by another local project (`paperless-receipt-automation`'s Vite), which made `npm run dev` fail (`strictPort`). TubeAtlas now uses 5171.
- Observed once: the launcher stopped the API while it was creating a fresh database (because Vite failed on the busy port). The result was an empty but consistent SQLite file that migrated normally on the next start. Migrations are transactional, so this is safe.
