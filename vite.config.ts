import { appendFileSync, createReadStream, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { resolve, sep } from 'node:path';
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
      // Chrome's Crash Reporting API: when the browser kills a page (reason "oom" for memory), it posts a report
      // here (see Reporting-Endpoints below). Kept in .devlog/crash-reports.jsonl.
      server.middlewares.use('/__crash-reports', (req, res) => {
        let body = '';
        req.on('data', (chunk: Buffer) => {
          body += chunk.toString('utf8');
          if (body.length > 100_000) req.destroy();
        });
        req.on('end', () => {
          mkdirSync('.devlog', { recursive: true });
          appendFileSync('.devlog/crash-reports.jsonl', JSON.stringify({ at: new Date().toISOString(), body }) + '\n');
          res.statusCode = 204;
          res.end();
        });
      });
    },
  };
}

/**
 * Development only: model files from the reference cache (tools/reference/.cache, filled by
 * tools/reference/fetch_model.py), so a dev browser loads models already on this Mac instead of
 * downloading them again. `/__models/<owner>__<repo>/<revision>/<path>`; a file not in the cache
 * redirects to Hugging Face. Ranges are honoured, as the app resumes downloads.
 */
function localModels(): Plugin {
  const root = resolve('tools/reference/.cache');
  return {
    name: 'quick-inf-local-models',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__models/', (req, res) => {
        const rel = decodeURIComponent((req.url ?? '').split('?')[0]!.replace(/^\//, ''));
        const file = resolve(root, rel);
        if (!file.startsWith(root + sep)) {
          res.statusCode = 400;
          res.end();
          return;
        }
        if (!existsSync(file)) {
          const [owner, revision, ...path] = rel.split('/');
          res.statusCode = 302;
          res.setHeader('location', `https://huggingface.co/${owner!.replace('__', '/')}/resolve/${revision}/${path.join('/')}`);
          res.end();
          return;
        }
        const size = statSync(file).size;
        const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
        const start = range ? Number(range[1]) : 0;
        const end = range && range[2] ? Number(range[2]) : size - 1;
        res.statusCode = range ? 206 : 200;
        res.setHeader('content-length', end - start + 1);
        res.setHeader('content-type', 'application/octet-stream');
        res.setHeader('accept-ranges', 'bytes');
        if (range) res.setHeader('content-range', `bytes ${start}-${end}/${size}`);
        if (req.method === 'HEAD') {
          res.end();
          return;
        }
        createReadStream(file, { start, end }).pipe(res);
      });
    },
  };
}

/**
 * Cross-origin isolation, so the page can use SharedArrayBuffer and ONNX Runtime's CPU engine can use several
 * threads (and Chrome's measureUserAgentSpecificMemory works). Every cross-origin request must then be CORS or
 * opt in with Cross-Origin-Resource-Policy; model downloads from Hugging Face are CORS fetches. Production (Vercel)
 * sends the same two headers from vercel.json.
 */
const ISOLATION_HEADERS = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
/** Where Chrome sends crash reports (development; production needs an endpoint of its own). */
const DEV_HEADERS = { ...ISOLATION_HEADERS, 'Reporting-Endpoints': 'crash-reporting="/__crash-reports"' };

export default defineConfig({
  plugins: [react(), devLog(), localModels()],
  worker: { format: 'es' },
  // Pre-bundle ONNX Runtime when the dev server starts, so a first load doesn't make Vite re-bundle mid-request.
  optimizeDeps: { include: ['onnxruntime-web/webgpu', 'onnxruntime-web/jspi', 'onnxruntime-web/wasm'] },
  server: {
    https,
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    headers: DEV_HEADERS,
  },
  preview: { headers: ISOLATION_HEADERS },
});
