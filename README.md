# TubeAtlas

A personal, local-first knowledge hub for YouTube videos: watch a video beside its timestamped transcript, explore its concepts as a knowledge graph, ask grounded questions, and keep notes.

TubeAtlas is being rebuilt as a Node.js/TypeScript application. The plan lives in [docs/implementation-plan.md](docs/implementation-plan.md), the knowledge-graph requirements in [docs/knowledge-graph-quality.md](docs/knowledge-graph-quality.md), and the step-by-step work in [docs/work-packages.md](docs/work-packages.md). Progress is recorded in [docs/work-log.md](docs/work-log.md). Features are documented here as they land.

## Available now

- Import a YouTube video by link or ID and see its transcript status update automatically.
- Browse your library; create, rename, and delete topics; assign videos to topics and filter by collection. Deleting a topic keeps its videos.
- Open a video overview with its metadata and an external YouTube link.
- Read configured models, key presence, the data directory, and backup instructions in Settings.
- Use the interface at desktop or phone width, including keyboard navigation.

Watch & Read is the next work package. The reader, notes, chat, and knowledge graph will appear as they are completed.

## Requirements

- Node.js 26.10 or newer (see `.nvmrc`). Node runs the server's TypeScript directly; there is no server build step.
- An [OpenRouter](https://openrouter.ai/) API key for AI features (optional to start the app).
- Optionally a YouTube Data API key, which adds video durations.

## Setup

```bash
cp .env.example .env   # then fill in your keys
npm ci
```

## Commands

| Command | What it does |
| --- | --- |
| `npm run dev` | API on 127.0.0.1:5170 (restarts on change) and the frontend on http://127.0.0.1:5171 |
| `npm run build` | Builds the frontend into `frontend/dist` |
| `npm start` | Serves the API and the built frontend on http://127.0.0.1:5170 |
| `npm run check` | Typechecks server and frontend, then builds |
| `npm test` | Runs the tests (no network access, no API cost) |

`better-sqlite3` ships prebuilt binaries, so its compile-from-source install script is denied in `package.json` (`allowScripts`). No C/C++ toolchain is needed.

## Data and backup

Everything you create is stored in `data/` (database and files); it is not tracked by git. To back up, stop the app and copy the `data/` directory.

The previous Python/FastAPI implementation was removed; it remains available in the git history.
