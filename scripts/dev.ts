// Runs the API (node --watch) and the Vite dev server together; stops both when either exits or on Ctrl+C.
import { spawn, type ChildProcess } from 'node:child_process';

const children: ChildProcess[] = ['dev:api', 'dev:web'].map((script) =>
  spawn('npm', ['run', '--silent', script], { stdio: 'inherit' }),
);

let stopping = false;
function stopAll(code: number) {
  if (stopping) return;
  stopping = true;
  for (const child of children) if (child.exitCode === null) child.kill('SIGTERM');
  process.exitCode = code;
}

for (const child of children) child.on('exit', (code) => stopAll(code ?? 0));
process.on('SIGINT', () => stopAll(0));
process.on('SIGTERM', () => stopAll(0));
