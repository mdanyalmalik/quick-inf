#!/usr/bin/env node
// Runs the speed bench page (bench.html) in Chrome for each model and prints the results as JSON lines.
// Needs the dev server running (npm run dev, or npx vite --port 5174); models come from its local cache
// (tools/reference/.cache), or Hugging Face when not there.
//
//   node tools/bench/bench.mjs [--origin https://localhost:5174] [--headed] [--log REGEX] [--diagnostics] model[:query] ...
//   e.g. node tools/bench/bench.mjs smollm2-360m-instruct 'ternary-bonsai-2-2b:tokens=64&prompt=500'
//
// --cool SECONDS waits between models (default 20), so one model's heat doesn't slow the next (fanless laptops).
// --profile FILE records a CPU profile of the run (Chrome's sampling profiler) to FILE (.cpuprofile).
// --chrome-flag FLAG passes a command-line flag to Chrome (repeatable), e.g. --enable-dawn-features=disable_robustness.
// --page NAME runs NAME.html instead of bench.html (e.g. kernels: the spec's query is then the page's whole query).
// --diagnostics runs diagnostics.html instead (correctness against the reference fixtures) and prints its checks.
// --log prints the page's console messages that match REGEX (e.g. ONNX Runtime's node placement, with
// session={"logSeverityLevel":0,"logVerbosityLevel":1} in the query). Uses a temporary Chrome profile, deleted afterwards.

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
let origin = 'https://localhost:5174';
let headed = false;
let logPattern = null;
let diagnostics = false;
let profileFile = null;
let coolSeconds = 20;
let pageName = 'bench';
const chromeFlags = [];
const models = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--origin') origin = args[++i];
  else if (args[i] === '--headed') headed = true;
  else if (args[i] === '--log') logPattern = new RegExp(args[++i]);
  else if (args[i] === '--diagnostics') diagnostics = true;
  else if (args[i] === '--profile') profileFile = args[++i];
  else if (args[i] === '--cool') coolSeconds = Number(args[++i]);
  else if (args[i] === '--page') pageName = args[++i];
  else if (args[i] === '--chrome-flag') chromeFlags.push(args[++i]);
  else models.push(args[i]);
}
if (!models.length) {
  console.error('Usage: node tools/bench/bench.mjs [--origin URL] [--headed] model[:query] ...');
  process.exit(1);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), 'qi-bench-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(
  CHROME,
  [
    ...(headed ? [] : ['--headless=new']),
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    '--ignore-certificate-errors',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    ...chromeFlags,
    'about:blank',
  ],
  { stdio: 'ignore' },
);

async function devtools() {
  for (let i = 0; i < 100; i++) {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
      const page = targets.find((t) => t.type === 'page');
      if (page) return page.webSocketDebuggerUrl;
    } catch {}
    await wait(100);
  }
  throw new Error('Chrome did not start');
}

function connect(url) {
  const ws = new WebSocket(url);
  let id = 0;
  const pending = new Map();
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.method === 'Runtime.consoleAPICalled' && logPattern) {
      const text = msg.params.args.map((a) => a.value ?? a.description ?? '').join(' ');
      if (logPattern.test(text)) console.log('[console]', text.slice(0, 2000));
    }
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result);
    }
  };
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const n = ++id;
      pending.set(n, { resolve, reject });
      ws.send(JSON.stringify({ id: n, method, params }));
    });
  return new Promise((resolve) => (ws.onopen = () => resolve({ send, close: () => ws.close() })));
}

try {
  const cdp = await connect(await devtools());
  if (logPattern) await cdp.send('Runtime.enable');
  const evaluate = async (expression) => (await cdp.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result.value;
  for (const [index, spec] of models.entries()) {
    if (index > 0 && !diagnostics) await wait(coolSeconds * 1000);
    const at = spec.indexOf(':');
    const [id, query] = at < 0 ? [spec, ''] : [spec.slice(0, at), spec.slice(at + 1)];
    const started = Date.now();
    if (profileFile) {
      await cdp.send('Profiler.enable');
      await cdp.send('Profiler.setSamplingInterval', { interval: 100 });
      await cdp.send('Profiler.start');
    }
    const page = diagnostics ? 'diagnostics.html' : `${pageName}.html`;
    const global = diagnostics ? 'window.__diagnostics' : 'window.__bench';
    const url = `${origin}/${page}?model=${encodeURIComponent(id)}${diagnostics ? '&autorun' : ''}${query ? '&' + query : ''}`;
    await cdp.send('Page.navigate', { url });
    let result = null;
    let reloaded = false;
    for (;;) {
      await wait(1000);
      // A page whose script never started (the dev server re-bundling dependencies mid-load): load it once more.
      if (!reloaded && Date.now() - started > 90_000 && !(await evaluate(`!!${global}`).catch(() => false))) {
        reloaded = true;
        await cdp.send('Page.navigate', { url });
      }
      result = await evaluate(`${global} && ${global}.done ? JSON.stringify(${global}) : null`).catch(() => null);
      if (result) break;
      if (Date.now() - started > 30 * 60 * 1000) {
        result = JSON.stringify({ error: 'timed out', log: await evaluate('window.__bench && window.__bench.log').catch(() => null) });
        break;
      }
    }
    if (profileFile) writeFileSync(profileFile, JSON.stringify((await cdp.send('Profiler.stop')).profile));
    const bench = JSON.parse(result);
    if (diagnostics) {
      const checks = (bench.results ?? []).map((r) => ({ name: r.name, outcome: r.outcome, ...(r.outcome === 'fail' || /Speed|Teacher|Greedy/.test(r.name) ? { details: r.details } : {}) }));
      console.log(JSON.stringify({ model: spec, s: Math.round((Date.now() - started) / 1000), error: bench.error, failed: checks.filter((c) => c.outcome === 'fail').length, checks }));
      continue;
    }
    console.log(JSON.stringify({ model: spec, s: Math.round((Date.now() - started) / 1000), ...(bench.error ? { error: bench.error, log: bench.log } : bench.results) }));
  }
  cdp.close();
} finally {
  chrome.kill();
  await wait(500);
  rmSync(profile, { recursive: true, force: true });
}
