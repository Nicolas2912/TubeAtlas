import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { DEV_WEB_PORT } from '../shared/ports.ts';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: DEV_WEB_PORT,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:5170' },
  },
  build: { outDir: 'dist', emptyOutDir: true },
});
