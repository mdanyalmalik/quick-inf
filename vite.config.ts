import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// Local HTTPS (mkcert) so phones on the same Wi-Fi get a secure context, which
// WebGPU requires. The certificate covers localhost, 127.0.0.1 and the Mac's
// LAN IP; regenerate it with mkcert if that IP changes (see .certs/, gitignored).
// Without the certificate files, the dev server falls back to plain HTTP.
const cert = '.certs/cert.pem';
const key = '.certs/key.pem';
const https = existsSync(cert) && existsSync(key) ? { cert: readFileSync(cert), key: readFileSync(key) } : undefined;

/**
 * Development only: devices on the LAN (e.g. a phone running the diagnostics
 * page) POST results to /__devlog; each is appended as one JSON line to
 * .devlog/devlog.jsonl (gitignored), so they can be read on the Mac. The phone
 * suite also reads remote commands from /__phone-control.
 */
function devLog(): Plugin {
  return {
    name: 'quick-inf-devlog',
    apply: 'serve',
    configureServer(server) {
      // Keep-alive for phones: the suite page pings this a few times a second so the phone's Wi-Fi
      // doesn't doze between requests (with power saving, responses stalled for up to a minute).
      server.middlewares.use('/__ping', (_req, res) => {
        res.statusCode = 204;
        res.setHeader('cache-control', 'no-store');
        res.end();
      });
      // Commands for the phone test suite, written on the Mac to .devlog/phone-control.json
      // ({ id, action: 'run' | 'clear' | 'reload', params }); the suite polls this while idle.
      server.middlewares.use('/__phone-control', (_req, res) => {
        const file = '.devlog/phone-control.json';
        res.setHeader('content-type', 'application/json');
        res.setHeader('cache-control', 'no-store');
        res.end(existsSync(file) ? readFileSync(file) : '{}');
      });
      server.middlewares.use('/__devlog', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end();
          return;
        }
        let body = '';
        req.on('data', (chunk: Buffer) => {
          body += chunk.toString('utf8');
          if (body.length > 1_000_000) req.destroy();
        });
        req.on('end', () => {
          mkdirSync('.devlog', { recursive: true });
          const entry = { at: new Date().toISOString(), from: req.socket.remoteAddress, userAgent: req.headers['user-agent'], body };
          appendFileSync('.devlog/devlog.jsonl', JSON.stringify(entry) + '\n');
          res.statusCode = 204;
          res.end();
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), devLog()],
  // Module workers, so the worker can load WebLLM (6.6 MB) as a separate chunk only when needed.
  worker: { format: 'es' },
  // Pre-bundle the engines when the dev server starts. Otherwise the first lazy import of WebLLM
  // makes Vite discover it, re-bundle, and reject the requests in flight ("Failed to fetch
  // dynamically imported module"), which failed the first WebLLM load on a freshly started server.
  optimizeDeps: { include: ['@mlc-ai/web-llm', 'onnxruntime-web/webgpu'] },
  server: {
    https,
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
  },
});
