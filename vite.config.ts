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
 * .devlog/devlog.jsonl (gitignored), so they can be read on the Mac.
 */
function devLog(): Plugin {
  return {
    name: 'quick-inf-devlog',
    apply: 'serve',
    configureServer(server) {
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
  server: {
    https,
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
  },
});
