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

## WP-04: Frontend shell, Library, Topics, Settings (2026-10-03)

This entry records the original implementation and verification. Mobile support and the associated acceptance requirements were subsequently removed at the user's request; see the desktop-only follow-up below.

**Done**
- Connected React Router, lazy views, and the shared styles to the application entry point. Completed the desktop shell, mobile menu, library, import dialog, topic management and filtering, Settings, and video overview.
- Corrected the typed API client: `AppType` is already the API sub-app, so the client uses `/api` directly. Added three tests that exercise the frontend client against the actual server, including imports, jobs, topic assignment/filtering/rename/delete, settings, API errors, and connection errors.
- Completed topic assignment from the video header, with saves reflected immediately and controls disabled during a save. Topic deletion asks for confirmation and keeps the videos. Empty, loading, missing-record, and recoverable-error states are implemented.
- Fixed stale query results on collection changes and stale job state on job changes. Closed import dialogs ignore late responses instead of unexpectedly navigating the user away.
- Added a real video overview with metadata, topic links, transcript job status, and an external YouTube link. `VideoLayout` supplies the header, breadcrumb, conditional tabs, and nested outlet context for subsequent video views.
- Settings shows key presence, configured models, the data directory, and backup/restore instructions without exposing keys. Updated README and the implementation/work-package plans to reflect what is available.

**Verification**
- `npm run check && npm test`: both typechecks and the production build pass; **67/67 tests pass**. No dependencies were added to the project.
- Browser automation used `npx --yes --package agent-browser agent-browser --session tubeatlas-wp04` after the T3 preview explicitly reported that no automation host was available. All browser data was stored in a fresh temporary `DATA_DIR`; existing application data and evaluation snapshots were left untouched.
- Real YouTube import at **1440×900** and a second fresh import at **390×844**, both through the dialog using `jGD_UR4wMJc`. A browser MutationObserver recorded `Fetching transcript…` followed by `Ready` without reloading, at both widths. The latest live job succeeded with **266 English segments** (YouTube captions have changed since WP-03's 261-segment snapshot; that snapshot was not overwritten).
- At both widths: assigned topics from the video-header picker, followed the topic link to the filtered library, reloaded, and saw the video with the saved assignments. An empty collection displayed no videos. A full dev-server shutdown/restart preserved the video, transcript status, topics, and counts.
- Topic management: created an empty topic, rejected a duplicate name inline, renamed it, and deleted a populated topic after confirming the message that its videos would remain. The library still contained its video afterwards.
- Import errors: an invalid link displayed `That is not a YouTube video link.`; `ZZZZZZZZZZZ` displayed `YouTube has no video with that id.` inside the dialog. A transient live metadata failure displayed `Could not reach YouTube.`; submitting again succeeded.
- Keyboard: Tab traversed the global navigation, import button, topic navigation, create/rename/delete controls, video title, and topic links, each with a visible solid focus outline. Enter opened the import dialog; Shift+Tab from its first field wrapped to the last button, Tab wrapped back, and Escape closed it and restored focus. The mobile menu also trapped focus, closed on Escape, and closed after navigation. Its hidden desktop counterpart did not appear in the narrow-width tab order.
- Axe accessibility checks reported **zero violations** for the tested Library, video, Topics, and Settings pages after correcting primary-button contrast and modal focus wrapping. Narrow Library, video, Topics, and Settings pages had no horizontal overflow.
- `npm start` served the built application from port **5170**. Direct navigation to `/topics/1`, `/videos/1`, and `/settings` rendered the saved data and styles. Re-importing the existing video kept the video count at one and retained both topic assignments. A deliberately aborted topic-list request showed the connection error; after restoring the network and clicking `Try again`, all three topic-navigation links returned and the error disappeared. Missing topic/video routes displayed useful messages; an unknown page rendered the 404 view inside the shell.
- Captured desktop and narrow screenshots (`/tmp/tubeatlas-wp04-{desktop,mobile-video,mobile-topics,mobile-settings}.png`) and compared the desktop shell with `output/ui-concepts/focused-views/06-watch-and-read.png`: 190 px sidebar, warm background/surfaces, serif headings, muted body text, and terracotta accents match the prescribed design tokens. The overview uses a 16:9 cover next to a white status panel, stacked on narrow screens. Primary buttons use the darker accent to meet contrast requirements. Shared tab CSS retains the accent and 2 px active underline; unfinished tabs are intentionally absent.

**Deviations and notes**
- WP-04 originally required importing into `/videos/:id/watch`, while explicitly excluding the reader and forbidding unfinished tabs. Imports/cards now open the functional `/videos/:videoId` overview. WP-05 will add the reader route/tab, switch those destinations to `/watch`, and redirect the bare video route. The implementation plan and WP-04 instructions were updated together to resolve this contradiction.
- The screenshot criterion now separates shell comparison from tab styling, because rule 7 forbids showing any of the four video tabs before its WP is complete. No later feature was implemented or presented as complete.
- Routing and lazy components live in `app/App.tsx`, separate from the React mount in `main.tsx`; the topic helper is separate from `ImportDialog.tsx`. This fixes the duplicate-root/DOM-removal errors encountered during development updates. The 404 route also supplies its error view and an initial loading fallback.
- The configured YouTube key was rejected during the live check, and metadata correctly fell back to oEmbed; duration was therefore unknown. Settings reports key presence, not provider validation. No OpenRouter calls were made and AI spend was **$0**.
- The commit was initially blocked by a generated Python `pre-commit` hook left in `.git/hooks/`, referencing the configuration removed in WP-00. After confirming that no current pre-commit configuration exists, retained it as `.git/hooks/pre-commit.legacy-python` and removed the obsolete hook from the active path. The project’s current typechecks, build, and tests had already passed.

## WP-04 follow-up: Desktop-only interface (2026-10-03)

**Done**
- At the user's request, removed the mobile menu, duplicated navigation, viewport listeners, menu icon, and mobile layout breakpoints. The desktop sidebar and pane layouts remain in place. Dialog and topic-picker widths are fixed for desktop use.
- Updated README, the implementation plan, and WP-04/WP-05/WP-13 instructions and acceptance criteria to desktop-only scope. Mobile layouts and phone-width testing require a future user request. The preceding work-log entry remains a historical record of the original verification.

**Verification**
- `npm run check` passes both typechecks and the production build; `npm test` passes **67/67 tests**. `git diff --check` passes. No dependencies or tests were added for this UI removal.
- Served the production build with `npm start` using the existing temporary WP-04 data directory. T3 preview inspection explicitly reported no automation host, so used `agent-browser --session tubeatlas-desktop` as the browser fallback.
- Checked Library, Topics, Settings, and the video overview only at **1440×900**. Clicked through desktop navigation, inspected screenshots, and confirmed one main navigation, the visible 190 px sidebar and search box, no mobile menu, no CSS media rules, and no horizontal overflow. Topics, Settings, and the video overview retain their desktop columns.
- The import dialog opens at 480 px; Shift+Tab wraps from the URL field to Import, Tab wraps back, and Escape closes it and restores focus to Import video. The topic picker opens at 340 px within the viewport, shows the saved topic assignments, and closes with Escape.
- The video overview's axe check reported **zero violations**. Browser errors and console output were empty. Screenshots are `/tmp/tubeatlas-desktop-only-{library,topics,settings,video}.png`. Closed the test browser and stopped the server afterwards.

## WP-05: Watch & Read (2026-10-03)

**Done**
- Added the desktop Watch & Read view with the real YouTube IFrame Player API beside a searchable transcript. Library cards and imports open `/videos/:id/watch`; the bare video route redirects there with its query parameters intact. Watch & Read is the only implemented video tab.
- Added source-preserving passage grouping, active-passage highlighting, follow playback, timestamp seeking, linked timestamps, periodic/pause/seek/leave playback saves, and text/Markdown exports. Search ignores case and diacritics while highlighting the original characters; previous/next controls navigate individual matches. Scroll gestures suspend automatic following for five seconds, and the follow toggle stops automatic scrolling.
- Added paste and VTT/SRT upload fallback, inline errors that retain the draft, and caption retry through the real job runner. A new transcript-retry operation handles blocked outcomes whose earlier job succeeded; active retries are reused and ready transcripts are protected.
- Added retry controls for player loading and playback saves. Both the shared API script and an unresponsive player have bounded waits; unavailable or embedding-disabled playback keeps the transcript and offers an external link at the current time. The shared size-limit constant is separate from Zod so the reader does not bundle request-validation code.

**Verification**
- Final `npm run check` passes both typechecks and the production build; `npm test` passes **73/73 tests**. Six new tests cover passage timings/gaps/overlaps, untimed text, URL/saved/clamped positions, Unicode search/highlight ranges, timestamp exports, and blocked-caption retry/reuse/protection. The existing frontend-client test now exercises transcript reads and persisted playback saves. All test content is invented; no dependencies were added.
- Used the production server with a fresh `/tmp/tubeatlas-wp05-bC3x86` data directory. T3 preview later explicitly reported no automation host; continued with `agent-browser --session tubeatlas-wp05`. Browser checks used **1440×900 desktop** dimensions. Existing application data and legacy files were untouched.
- Imported `jGD_UR4wMJc` through the dialog with the live adapter, obtaining **266 English segments** and opening the reader. Opened `?t=300`: both the player start parameter and the reader position were 300 s. A timestamp click sought the real player to about 62.6 s for the 1:01 passage (YouTube seeks to nearby keyframes).
- A passive observer of real YouTube playback messages checked the visible position and highlighted passage every second for **32 samples / 30.81 s of actual playback**, with no mismatches. The active passage changed from 295.68 s to 318.16 s and the transcript scroll moved from 2352 to 2649 px. Saved positions advanced about every ten seconds. Turning follow off kept a manually reset scroll position at zero after seeking.
- Paused through the native player: actual and saved positions were both **128.742249 s**. Reopening the canonical reader and reloading both started at 128 s. The bare `/videos/5?t=300` route redirected to `/videos/5/watch?t=300` and retained the 300 s player start.
- German transcript: **316 segments / 32 passages**, intact umlauts; `ki` returned **19 matches**, including uppercase `KI`; `fur` matched accented text. No-match search disabled previous/next. Clearing a search using native Backspace restored all passages. Tab reached the scroll region and timestamp buttons with solid **2 px** focus outlines.
- Long transcript: **4,841 segments / 522 passages**. Reader mount to all passages took **98.5 ms**; searching `the` took **57.9 ms**, with **2,516 matches** in 509 passages. Next selected match 2 with one selected highlight. Twenty scroll/frame samples across the 509 matching passages had **17.3 ms p95 / 17.4 ms max**; a fresh check of all 522 passages measured **28 ms p95 / 37.4 ms max**. A separate Node measurement took about 5.9 ms to group, 27.2 ms to index, and 2.7 ms to search. No virtualization was needed.
- Browser-downloaded exports were compared byte-for-byte with the full stored transcript's grouped passages: **10,133 bytes text / 11,059 bytes Markdown**, with all 31 English passages and valid reader timestamp links.
- Seeded temporary missing-caption cases: pasted text became two untimed paragraphs, with no timestamp buttons and follow disabled, and survived reload. Deliberately aborting its save showed the connection error and retained the draft; restoring the connection and saving succeeded. An invalid VTT showed `No captions found in that file.`; valid VTT and SRT uploads retained starts `[1,10]` and `[2,12]`, respectively, and survived reload.
- Simulated blocked captions in the temporary German record: an aborted retry showed the inline connection error; a subsequent live retry returned **316 segments, revision 2** and replaced the fallback with the reader. Aborting playback saves showed the unsaved-position message; retrying after restoring the connection persisted the position.
- Aborted the actual YouTube embed request: after the 15 s deadline, the player offered retry while all 522 long-video passages remained readable and seeks were disabled. Restoring the request and clicking Try player again restored one working frame and enabled seeking. A caption-upload test video whose player rejected embedding displayed `This video can't be played here.` and an external link with the current timestamp; its transcript stayed readable.
- Stopped and restarted the server with the same temporary data directory. The English transcript and current playback position (208.953061 s at this later check, player start 208 s), pasted text, and timed uploads persisted. Ordinary checks after clearing deliberate-failure diagnostics produced no browser errors or console output.
- Axe reported **zero violations in the owned transcript panel**, including untimed text. The full-page audit identified two ARIA violations inside YouTube's third-party frame, outside TubeAtlas's control. Offscreen passages required manual contrast review; the reader's text/timestamp/highlight colour pairs range from **4.82:1 to 17.29:1**. Screenshots were inspected at `/tmp/tubeatlas-wp05-{watch,paused,german,long,long-search,untimed,upload,final}.png` against the desktop design reference. Closed the test browser and stopped the server afterwards.

**Transcript provenance**
- Re-fetched missing snapshots on 2026-10-03 with `node scripts/fetch-transcript.ts <id>`; saved under ignored `data/evaluation/`. No existing snapshot was overwritten. German and long-video browser data were loaded from these authentic snapshots; the English browser import used live captions.
- English `jGD_UR4wMJc`, en, 266 segments: SHA-256 `bb54465772147dde43101c3005ec232272aac6f6ad41808ac4d82349db9a9b3e`.
- German `Vzaccv7-qNw`, de, 316 segments: SHA-256 `c536ecba5097f3e7a9c82ee22f19cf963b8e5be0b3ab49088f012065d823c906`.
- Long `KIY0np5KDfE`, en, 4,841 segments: SHA-256 `d0faeb8e0e9f8e742eddf0585348196575b8b371a8c00b5b01dadf11da5f18e6`.

**Deviations and notes**
- Omitted deprecated `modestbranding`; supplied the page origin as recommended by the [official YouTube player parameters](https://developers.google.com/youtube/player_parameters). The [official API reference](https://developers.google.com/youtube/iframe_api_reference) documents seeking/keyframe behavior and player errors. Player polling is 250 ms, and writes are serialized to keep older requests from overwriting a newer pause or seek.
- Selection actions remain hidden until WP-06/WP-08 implement note saving and asking. Source timings are retained; only playback targets are clamped. A valid timestamp query intentionally takes precedence over the saved position on load.
- The configured YouTube key was rejected and metadata used oEmbed, so duration was unknown in the live imports. The player still clamps seeks using its duration when available. No OpenRouter calls were made; AI spend was **$0**.

## WP-06: Documents and attachments (2026-10-03)

**Done**
- Added document CRUD and server-side Markdown append/export, using the existing document and asset tables. Editable kinds are Note, Summary, Study guide, and Q&A; file attachments can be renamed but cannot be edited as text or changed into a note.
- Added multipart uploads with extension and signature checks. PDF, PNG, JPEG, and WebP preview; DOCX/XLSX/PPTX download; valid UTF-8 Markdown/text becomes a note. Files have generated storage names, streamed content, safe download filenames, `nosniff`, and sandbox headers. Failed inserts roll back and clean files; failed database deletions restore the staged file.
- Added the lazy desktop Documents tab: 28% list, filtering/search, New/Upload, serif title, formatting commands, Edit/Preview, export, rename/kind/delete options, source footer, and saved edit time. Autosave waits 800 ms, runs one request at a time, and immediately follows with edits made during a save. Failed drafts remain editable with a retry button; router and page-leave protection ask before discarding changes.
- Connected Save to note from transcript selections. It captures paragraph text without timestamp buttons, creates a new note or appends to this video's editable document, preserves the first selected passage's source link, and omits timestamps for untimed text. The popover supports keyboard focus and Escape. Ask AI and later views remain hidden.

**Verification**
- Final `npm run check && npm test` passes both typechecks, the production build, and **80/80 tests**. Seven new tests cover document CRUD/append/export, file byte integrity and headers, rejected signatures/HTML/SVG/invalid text, text imports, body limits, insert/delete rollback, formatting, and timed/untimed quotes. The existing frontend-client test additionally exercises real document writes and multipart uploads. All unit-test text and files are invented; no dependency or migration was added. The intentional SQLite-failure tests produce expected 500 diagnostics and verify recovery.
- Used a fresh temporary data directory, `/var/folders/9d/t0mzs6d12wl5mvtq2crq0wb00000gn/T/tubeatlas-wp06-IyEO9t`, with production `npm start` on 5170 and a development check on 5171. T3 preview initialization was attempted; resize timed out and snapshot explicitly reported no automation host, so browser checks used `agent-browser --session tubeatlas-wp06`. All visual checks used **1440×900 desktop** dimensions. Existing app data, evaluation snapshots, and legacy artifacts were untouched.
- Created a note through New, entered a title and Markdown, waited for Saved, then reloaded: both survived. Stopped the actual server, typed a new draft, saw Save failed and the retained text, restarted, and typed one character: Saved returned and the database contained the entire draft. Separately aborted a save and verified that Retry save persisted the retained draft after restoring the request.
- Delayed the response from a real PATCH by 1,200 ms and edited during that save. Two writes began at **811.4 ms** and **2,023.2 ms** after the audit started, with **one maximum active request**. The second contained the latest edit and followed the first immediately; the stored content matched it. Repeated a save under development React Strict Mode and confirmed Saved plus the updated list.
- With a failed draft, dispatching a cancellable `beforeunload` event was prevented. Clicking Watch & Read opened the real unsaved-changes confirmation; cancelling retained the route and draft, and confirming a subsequent leave discarded only those unsaved edits.
- Used the formatting toolbar on a selected phrase and another selected line; Bold and Heading 2 rendered as strong text and a heading in Preview. Renamed through the options menu to a title containing `Ü`, changed the kind to Study guide, and saw both changes persist. Notes/Files filters showed two editable documents and three attachments; title search found the Unicode title, and clearing it restored all five rows.
- Uploaded a disguised HTML `.pdf`: inline error, existing document retained. Uploaded a real **240×140 PNG** and a one-page PDF; the image and browser PDF viewer rendered their content. Uploaded a real minimal DOCX and downloaded it through the UI; SHA-256 matched the original: `3060af05f5ab29925189f9338fa89bd94776841c4e93d49e9184e9b92889ea53`. A Markdown upload became an editable note.
- Downloaded Markdown exports through the browser and compared them with the saved content byte-for-byte, including a final **436-byte** export containing the appended source. Exporting an attachment downloads its original format.
- Selected **344 characters** from a timed passage starting at **61.64 s**. An aborted append retained the selection and showed a connection error; restoring the request and clicking the target again preserved the existing note and appended the quote. Opened that note, clicked its rendered **1:01** source link, and confirmed `?t=61`, the native iframe's start 61, the visible position 1:01, and a real YouTube playback message reporting **61 s**.
- Selected across two invented untimed paragraphs and created a new note. It preserved both paragraphs, contained no timestamp link, and survived reload. The target list contained editable notes and excluded all attachments.
- Restarted the server with the same temporary directory: the timed quote, renamed/kind-changed note, PDF, and PNG remained available. On the DOCX deletion confirmation, Cancel kept the document and downloadable file; confirming removed its document and asset (both 404), removed the managed file, and returned to the empty editor. The files directory then contained only the PDF and PNG, with no temporary files.
- Keyboard: title → Markdown used a visible **2 px** focus outline. Enter opened Save to note with focus inside, Tab reached New note with the same outline, and Escape returned focus to Transcript passages. Axe reported **zero violations and zero incomplete checks** in the Documents view; the owned transcript panel also had zero violations. Desktop pages had no horizontal overflow. Ordinary browser errors/console were empty after clearing intentional network-failure diagnostics.
- Inspected screenshots `/tmp/tubeatlas-wp06-{editor,save-failed,preview,png,pdf,save-to-note,final-editor}.png` against `10-documents.png`. Closed the test browser and stopped both servers afterwards.

**Deviations and provenance**
- Document POST/PATCH use a 25 MB request limit so uploaded text notes can be edited, alongside the specified 25 MB multipart limit. Envelopes count toward the request limit; accumulated Markdown is capped at 25 MB of UTF-8. Attachment export uses `?download=1`; text export waits for autosave to finish.
- Reused the authentic English live-import transcript from WP-05 read-only, copied into the temporary test database: `jGD_UR4wMJc`, en, 266 segments, acquired 2026-10-03. Its stored `segments_json` SHA-256 is `73381fb3f628c8c39c06bec84d4c6d13c2c71170a1d0fc244bd40c04a155a8b3`. No evaluation transcript content was added to code or tests. Untimed text and file fixtures were invented.
- No OpenRouter calls were made; AI spend was **$0**. No later work package was started.

## WP-07: Evidence units and retrieval (2026-10-03)

**Done**
- Added one pure, deterministic evidence-unit builder and shared version/types/token estimate. It validates times and source IDs, keeps raw segments immutable, normalizes displayed whitespace/NFC while preserving punctuation, moves non-speech tags into annotations, records removed rolling prefixes, and splits every speaker marker. Sentence punctuation closes units after 15 words; long sentences close at caption boundaries after 45 words. IDs remain unique beyond `u999`.
- Transcript GET and manual-save responses now include units, version 1, and deduplication provenance alongside original segments. Invalid caption ranges are rejected before replacing a usable revision.
- Added revision-scoped, on-demand retrieval using the existing chunks table and OpenRouter adapter. Consecutive whole units pack to about 350 estimated tokens; embedding batches contain at most 64 chunks. Configured model/version key the persistent cache. Builds share concurrent work, validate/normalize vectors, and atomically replace old chunks only after every batch succeeds. Retrieval scores normalized vectors, adds immediate neighbours, deduplicates, orders by time, and cuts at a whole-unit text-token budget. Reported costs and usage are logged without query/transcript text.
- Switched the desktop reader to unit search and playback highlighting, retaining paragraph typography, source timestamps, full exports, and note saving. Paragraphs break at turns or about 60 words, preserving untimed source paragraphs. Quotes use the first selected unit's time. Following centers multiline unit bounds; timestamp clicks explicitly resume centering.

**Verification**
- Final `npm run check && npm test` passes both typechecks, the production build, and **94/94 tests**. Fourteen new tests use invented text and cover deterministic preprocessing, nested overlapping times, mid-caption/mid-word markers, annotations, rolling-prefix mapping and no-op cases, sentence/length boundaries, null timing, normalization, validation, and IDs through `u1234`; retrieval covers revision/video isolation, cosine ranking, neighbours, budgets, stored float32 normalization, cache reuse/model/version rebuilds, 64/64/2 batching, packing, shared concurrent builds, failed-build recovery, invalid vectors, dimension mismatch, missing/deleted revisions, empty inputs, and known/unknown costs. Existing API/client tests verify returned units and that invalid uploads leave the current transcript intact. Expected SQLite-failure diagnostics from the earlier attachment tests remain deliberate.
- Verified the whole reader story: immutable transcript → derived API units → rendered paragraphs/search/highlight → saved quote/source link. Retrieval was verified from a real SQLite revision through the real OpenRouter adapter with a simulated HTTP response, persisted vectors, and returned units. No unit-test network calls or evaluation-text fixtures were added.
- Used production `npm start` on 5170 and isolated data `/tmp/tubeatlas-wp07-5aQfn7`. Attempted T3 preview status/open/resize/snapshot; snapshot explicitly reported no automation host, so browser checks used `agent-browser --session tubeatlas-wp07`, always **1440×900 desktop**. The shared preview was moved to Library after unexpected audible playback. Restarted the test browser with `--mute-audio`; YouTube messages also confirmed `muted: true`. All subsequent playback checks were muted, and the browser and server were closed afterwards.
- The short English reader showed all **75** API units byte-for-byte in **26** paragraphs. Search produced **94** expected matches in **53** matching units, and Next moved from `u001` to `u002`. A search-filtered text download still contained the full transcript: **10,128 bytes**, identical to the export generated from all units.
- Seeking the fourth paragraph selected `u009`, start **83.36 s**, with background on that unit only. Actual YouTube messages reported **84.099956 s** and subsequent playback moved the highlight through `u010`–`u018`. Following centered the later unit within **0.375 px**; the final explicit-seek check centered `u009` within **0.25 px**. Fixed centering to use multiline bounding height and reset the manual-scroll pause on an explicit seek.
- Selected **173 characters** from `u002` in the first paragraph: its source starts **5.36 s**, while the paragraph starts at **0 s**. Save to note persisted the exact quote with `/videos/2/watch?t=5`. Clicking its rendered source opened that URL with native iframe `start=5`.
- German search without the diacritic found **13/13** expected matches, all preserving the original accented highlights. A nonexistent word showed No matches, zero visible units, and disabled match navigation.
- The long reader rendered **1,472** units in **795** paragraphs, all text identical to the API; every non-empty turn started a paragraph and no tag/marker text leaked into speech. The observed full render completed at **1,104 ms**, with an additional transcript API read taking **67 ms**. Search yielded **2,511** matches in **1,020** units; Previous from the first match wrapped to the last, in `u1468`. No horizontal overflow occurred.
- Pasted invented untimed text through Add transcript, including decomposed Unicode, curly quotes/dash, a tag, and a speaker marker. Two raw paragraphs became three units/paragraphs with null times, one turn, and the tag retained only as an annotation. Canonical quote/dash search highlighted the original punctuation. Seeking/timestamps were absent and Follow was disabled. Saving a selection across all three paragraphs retained every unit and omitted a source timestamp; Escape restored transcript focus.
- Axe found **zero violations and zero incomplete checks** in the untimed transcript panel. The full timed panel had zero violations and an incomplete contrast check for clipped rows; computed contrast was **17.29:1** for normal unit text, **14.31:1** for active unit text, and **5.83:1** for timestamps. Ordinary browser errors/console were empty. Inspected `/tmp/tubeatlas-wp07-{unit-highlight,final-highlight,no-matches,long-search,untimed,after-restart}.png`.
- Stopped and restarted the server with the same data directory: all four transcript APIs returned 200 with the same unit counts and unchanged raw-segment hashes; the timed source and untimed note survived. Reloaded the untimed reader and saw all three units/paragraphs. Separately built one chunk from the invented untimed revision in one Node process, then reopened the database in a second process with an embedder that throws if called: all three units loaded from the durable cache without embedding.

**Snapshot measurements and provenance**

Read existing snapshots only; no sentences, names, aliases, or paraphrases entered code, prompts, or test fixtures. Snapshots were acquired on 2026-10-03 as recorded in WP-05. Converted the snapshot's `start + duration` representation into the application's seconds-based `start/end` in memory, rounding to millisecond precision. The files and their hashes stayed unchanged.

| Snapshot | Language | Segments | Units | Maximum words / characters | `turnStart` units | Removed prefixes / annotations |
| --- | --- | --- | --- | --- | --- | --- |
| `Vzaccv7-qNw` | de | 316 | 83 | 49 / 284 | 0 | 0 / 0 |
| `jGD_UR4wMJc` | en | 266 | 75 | 47 / 258 | 0 | 0 / 0 |
| `KIY0np5KDfE` | en | 4,841 | 1,472 | 53 / 307 | 509 | 10 / 31 |

- SHA-256, German: `c536ecba5097f3e7a9c82ee22f19cf963b8e5be0b3ab49088f012065d823c906`.
- SHA-256, short English: `bb54465772147dde43101c3005ec232272aac6f6ad41808ac4d82349db9a9b3e`.
- SHA-256, long English: `d0faeb8e0e9f8e742eddf0585348196575b8b371a8c00b5b01dadf11da5f18e6`.
- On both short snapshots, joining unit text exactly equals joining the NFC/whitespace-normalized input; rolling deduplication is a no-op. The long snapshot has 512 markers, with three empty turns, hence 509 non-empty `turnStart` units. Maximum lengths above 45 words are expected because length-only breaks occur at caption boundaries.

**Refinements for later WPs**
- Corrected the KG spec's "last segment end" to the maximum covered end, matching WP-07 and preserving nested overlaps. Clarified raw/display/matching text, conservative multiword rolling deduplication, empty turns, and untimed paragraph preservation in both documents. Added WP-05 as a dependency because WP-07 changes the reader.
- Chunk replacement happens after successful embedding rather than deleting a usable cache before a potentially failing paid call. The existing UNIQUE/FK constraints and transaction provide atomic replacement; no migration or dependency was added.
- A single unit larger than the chunk target remains whole. Retrieval's budget counts `estimateTokens(unit.text)`; WP-08 must reserve space separately for citation headers, instructions, documents, and the question. No public paid retrieval endpoint or embedding on page load was added; Chat will call `createRetrieval` on demand.
- No OpenRouter network calls were made; AI spend was **$0**. Existing app data and unrelated legacy artifacts were untouched. No later work package was started.
