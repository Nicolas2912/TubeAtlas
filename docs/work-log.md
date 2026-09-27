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

## WP-02: OpenRouter adapter and provider check (2026-09-27)

**Done**
- `server/src/integrations/openrouter.ts`: native-fetch adapter with `chat`, `chatStream`, `embed`, and `structured`, plus an exported `sseData()` parser.
  - One timeout per call covers both the request and reading the body: chat and stream 120 s, embeddings 60 s, structured 600 s; overridable for tests.
  - No retries. Caller aborts propagate as `AbortError`; everything else becomes a typed `AppError`, and messages never contain the key or request bodies.
  - `structured` sends exactly the specified body (explicit `reasoning.effort` with `exclude`, `provider.require_parameters`, `seed`, `max_tokens`, strict `json_schema`; no `temperature`/`top_p`). It refuses truncated, filtered, refused, wrong-model, or non-JSON answers.
- `scripts/check-providers.ts` (`npm run check:providers`): four live calls; prints model, provider, latency, cost, and details; exits 1 on any failure. The structured probe uses an invented book record and validates the answer with Zod.

**Verification**
- `npm run check` passes; `npm test` 31/31 pass. The 12 new adapter tests cover:
  - SSE: comments, pieces split mid-line and mid-CRLF, multi-line events, a final unterminated event, and multi-byte UTF-8 split across reads;
  - streaming deltas in order with model, provider, usage, and finish reason kept, and nothing read after `[DONE]`;
  - a mid-stream error (`PROVIDER_ERROR`, after the partial delta was delivered);
  - a caller abort (`AbortError`, deltas kept);
  - timeouts before and during streaming;
  - HTTP 429/503 (retryable) and 400 (not retryable, provider message included);
  - network failure, and a 200 with an error body;
  - the exact `structured` body with no `temperature`/`top_p`;
  - refusal, `length`, `content_filter`, model mismatch, and non-JSON answers;
  - embeddings reordered and normalized, and invalid, zero, inconsistent, or missing vectors rejected;
  - no call for empty input;
  - every failure path makes exactly one call, and no error message contains the API key.
- Live (`npm run check:providers`, 2026-09-27):

  ```text
  PASS chat       openai/gpt-4.1-mini via Azure, 1034 ms, cost $0.000008 — reply "OK"
  PASS chatStream openai/gpt-4.1-mini via Azure, 1119 ms, cost $0.000008 — reply "OK" in 1 delta(s)
  PASS embed      text-embedding-3-small via n/a, 225 ms, cost $0.000000 — 2 vectors × 1536 dims, norms 1.000/1.000
  PASS structured openai/gpt-6-astra via Azure, 1569 ms, cost $0.002040 — effort low, reasoning tokens 0, {"title":"The Lighthouse Keeper's Almanac","year":1897,"author":null}
  4/4 passed; reported cost $0.002056
  ```

- Failure path, at no cost: an invalid key gives 4 × `FAIL … PROVIDER_HTTP_401: OpenRouter returned HTTP 401: User not found.` and exit code 1; a missing key prints a pointer to `.env.example` and exits 1.

**Deviations and notes**
- `embed` returns `{ vectors, model, usage }` instead of bare vectors, because WP-07 must log embedding cost. `chat`/`chatStream` also return `latencyMs`. work-packages.md was updated.
- The embeddings endpoint reports the model as `text-embedding-3-small` (no `openai/` prefix), so WP-07 must key stored vectors on the configured model name. Noted in WP-02's as-built notes.
- Live spend for this WP: $0.002056.

## WP-03: YouTube import, transcripts, job runner (2026-09-27)

**Done**
- `server/src/integrations/youtube.ts`: `parseYouTubeId`, `parseIsoDuration`, `toSegments`, `cleanText`, and `createYouTube({ apiKey?, fetch?, sleep? })`.
  - Metadata comes from the Data API when a key is set, falling back to oEmbed on a key or quota error; oEmbed is used without a key.
  - Captions go through `youtube-transcript@1.3.1` with a recording `fetch`. It adds a per-request timeout and the job's abort signal, detects srv3 (ms) vs classic (s) from the caption XML, and notices 429s.
  - Outcomes: `ready`, `no_captions`, `blocked`, `failed`. Network errors are retried twice (1 s, 3 s).
- `server/src/jobs.ts`: persisted sequential runner (`enqueue` with duplicate protection, `cancel`, `retry`, `cancelForVideo`, `start`, `stop`, `idle`).
- Services:
  - `services/import.ts`: import flow and transcript job handler;
  - `services/transcripts.ts`: revisions; text, VTT, and SRT parsing;
  - `services/videos.ts`: list, get, update, delete, with active job and topics in summaries;
  - `services/topics.ts`.
- Routes: `routes/videos.ts`, `transcripts.ts`, `topics.ts`, `jobs.ts`; `server/src/validate.ts`; request schemas in `shared/api.ts`.
- Per-route body limits: 1 MB default, 5 MB for manual transcripts.
- `scripts/fetch-transcript.ts` (refuses to overwrite a snapshot).

**Verification**
- `npm run check` passes; `npm test` 63/63 pass (32 new):
  - URL parsing: 13 accepted forms and 9 rejects, including a look-alike host;
  - ISO durations;
  - srv3 ms and classic s conversion through the full adapter with invented XML;
  - an unknown caption format fails loudly;
  - 429 and captcha → `blocked`; no tracks → `no_captions`; unavailable → `failed`;
  - network retries (1 s, 3 s, then `failed`), recovery on the second attempt, and a caller abort is not retried;
  - metadata: Data API, not found, oEmbed fallback, 404/401/offline;
  - API: import → job → timed segments reload identically; re-import returns 200 without new calls; validation, `UNKNOWN_TOPIC`, `VIDEO_NOT_FOUND`; topics on import;
  - `no_captions`/`blocked` give a succeeded job and the right status;
  - a failed job → retry → success, and retrying a succeeded job → 409;
  - duplicate enqueue returns the active job; cancelling queued and running jobs;
  - manual text, VTT (header, NOTE, cue settings, inline tags, hour timestamps, empty cue), and SRT (BOM, CRLF, tags) create revisions 1–3 with only the newest current;
  - a 1.3 MB manual transcript is accepted, while 5 MB+ and a 1 MB+ topic body are refused;
  - list, topic filter, PATCH, and validation; topic CRUD with case-insensitive duplicates;
  - deleting a video removes its files, cancels its job, and returns 404 afterwards.
- Live (free), on a temporary `DATA_DIR`, with the real server, importing by URL (three different link forms):

  | Video | Import | Metadata (Data API) | Transcript |
  | --- | --- | --- | --- |
  | Vzaccv7-qNw | 201 | "Paperclip ist NEXT LEVEL!!", Niklas Steenfatt, 590 s | ready, de, 316 segments, last end 591.24 s |
  | jGD_UR4wMJc | 201 | "8 Jev Use Cases That Feel Like Cheating", Matthew Berman, 638 s | ready, en, 261 segments, last end 639.6 s |
  | KIY0np5KDfE | 201 | "Joe Rogan Experience #2553 - Andrew Huberman", PowerfulJRE, 9,599 s | ready, en, 4,841 segments, last end 9,592.64 s |

  Every stored transcript is **identical to its evaluation snapshot**, segment by segment (id, start, duration within 1 µs, and text). Re-importing returned 200. The server stopped cleanly with SIGINT (exit 0).
- oEmbed path without a key (live): title, channel, and thumbnail returned, duration `null`; an unknown id → `404 VIDEO_NOT_FOUND`.
- `node scripts/fetch-transcript.ts jGD_UR4wMJc` refuses to overwrite the existing snapshot (exit 1).

**Deviations and notes**
- Bug found by a test and fixed: an unreadable caption format made the package return `[]`, which was first reported as `no_captions`. The format is now checked first, so a YouTube format change fails loudly.
- Body limits are now per route (`bodyLimits()` in `app.ts`) instead of a single 1 MB limit on the whole API sub-app. WP-01's as-built note was corrected.
- Added, not in the original spec (all recorded in work-packages.md): oEmbed fallback when the Data API rejects the key or quota; `422 VIDEO_RESTRICTED`; `language` on manual transcripts; `activeJob` in video summaries; topic linking on re-import; `409 JOB_NOT_RETRYABLE`; runner `stop()` for tests and shutdown (waits at most 3 s); a pasted transcript isn't downgraded by a later failed fetch.
- The job runner's first loop tick is scheduled with `setImmediate`. Tests must stop the runner before closing the database; `createTestApp().cleanup` does this.
