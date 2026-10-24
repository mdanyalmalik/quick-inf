#!/usr/bin/env node
// Tests sharing a model across devices in the app itself (src/distributed, src/ui/share): the app as host turns
// sharing on, a window per device opens its join link, the host loads the model shared and chats; then one device
// closes, which must fail the next reply as a disconnection, and loading again shares it among the rest. Drives the
// app through its development hooks (window.__quickInf, window.__quickInfJoin), with a screenshot of the host's share
// panel. Needs the dev server (npx vite --port 5174); uses a temporary Chrome profile, deleted afterwards.
//
//   node tools/bench/share-app.mjs [--headed] [--devices N] [--mem GB] [--screenshot FILE] [--alone] [model]
//
// --mem gives every window that much measured memory (as if its benchmark had held it), so a model bigger than the
// cautious unmeasured share can be planned.

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const ORIGIN = 'https://localhost:5174';
const args = process.argv.slice(2);
let headed = false;
let devices = 2;
let screenshot = null;
let memGb = 0;
let alone = false;
let model = 'llama-3.2-3b-instruct';
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--headed') headed = true;
  else if (args[i] === '--devices') devices = Number(args[++i]);
  else if (args[i] === '--screenshot') screenshot = args[++i];
  else if (args[i] === '--mem') memGb = Number(args[++i]);
  else if (args[i] === '--alone') alone = true;
  else model = args[i];
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), 'qi-share-'));
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
    '--disable-backgrounding-occluded-windows',
    '--window-size=1280,900',
    'about:blank',
  ],
  { stdio: 'ignore' },
);

async function targets() {
  for (let i = 0; i < 100; i++) {
    try {
      return await (await fetch(`http://127.0.0.1:${port}/json`)).json();
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

const evaluate = async (cdp, expression) => {
  const { result, exceptionDetails } = await cdp.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
  return result.value;
};
const until = async (cdp, expression, what, seconds = 120) => {
  for (let t = 0; t < seconds * 2; t++) {
    if (await evaluate(cdp, expression).catch(() => false)) return;
    await wait(500);
  }
  throw new Error(`timed out waiting for ${what}`);
};
const log = (line) => console.log(`[${new Date().toISOString().slice(11, 19)}] ${line}`);

const reply = (cdp, content) =>
  evaluate(
    cdp,
    `(async () => {
      const c = window.__quickInf.client;
      try {
        const r = await c.generate({ turns: [{ role: 'user', content: ${JSON.stringify(content)} }], options: { maxNewTokens: 48, temperature: 0, topP: 1, repetitionPenalty: 1, repetitionWindow: 64, contextTokens: 4096 } }, () => {}).result;
        return { ok: true, text: r.text, tokPerSec: r.stats.decodeTokensPerSecond };
      } catch (e) {
        return { ok: false, code: e.code, message: e.message, status: c.getStatus?.().state };
      }
    })()`,
  );

try {
  // (A block, so --alone can stop early: `return` isn't allowed at the top level.)
  test: {
  const first = (await targets()).find((t) => t.type === 'page');
  const host = await connect(first.webSocketDebuggerUrl);
  // Every window (one browser profile) reads its memory from what the benchmark saved.
  if (memGb > 0) {
    await host.send('Page.navigate', { url: `${ORIGIN}/favicon.svg` });
    await until(host, 'document.readyState === "complete"', 'a page of the app’s origin', 30);
    await evaluate(host, `localStorage.setItem('quick-inf:memory-evidence', JSON.stringify({ failures: [], measuredBytes: ${memGb * 1e9} }))`);
  }
  // The model's own page: its load panel shows the share option and the room.
  // (A trailing slash: the dev server takes "/…-v0.2" for a file.)
  await host.send('Page.navigate', { url: `${ORIGIN}/${model}/` });
  // A first load can stall while the dev server bundles (after code changes): load it once more if so.
  await until(host, '!!window.__quickInf', 'the host app').catch(async () => {
    log('the host app didn’t start; loading it again');
    await host.send('Page.navigate', { url: `${ORIGIN}/${model}/` });
    await until(host, '!!window.__quickInf', 'the host app');
  });
  // --alone: the same questions with the model loaded here only, to tell a sharing fault from the model's own answer.
  if (alone) {
    const loaded = await evaluate(host, `window.__quickInf.client.load(${JSON.stringify(model)}).then(() => 'ok', (e) => e.code + ': ' + e.message)`);
    log(`loaded alone: ${loaded}`);
    for (const q of ['What is the capital of France?', 'Name a yellow fruit.']) log(`alone reply: ${JSON.stringify(await reply(host, q))}`);
    break test;
  }
  await evaluate(host, 'window.__quickInf.sharing.start()');
  await until(host, '!!window.__quickInf.sharing.get().getSnapshot().joinUrl', 'the room');
  const joinUrl = (await evaluate(host, 'window.__quickInf.sharing.get().getSnapshot().joinUrl')).replace(/^https?:\/\/[^/]+/, ORIGIN);
  log(`room open: ${joinUrl.replace(/#.*/, '#…')}`);

  const windows = [];
  for (let i = 0; i < devices; i++) {
    const { targetId } = await (await connect(first.webSocketDebuggerUrl)).send('Target.createTarget', { url: joinUrl, newWindow: true });
    const target = (await targets()).find((t) => t.id === targetId);
    const cdp = await connect(target.webSocketDebuggerUrl);
    await until(cdp, "window.__quickInfJoin && window.__quickInfJoin.getSnapshot().phase === 'connected'", `device ${i + 1} to connect`);
    const code = await evaluate(cdp, 'window.__quickInfJoin.getSnapshot().matchCode');
    windows.push({ targetId, cdp, code });
    log(`device ${i + 1} connected, match code ${code}`);
  }
  await until(host, `window.__quickInf.sharing.get().getSnapshot().devices.filter((d) => d.device).length === ${devices}`, 'the devices to say hello');
  const hostCodes = await evaluate(host, 'window.__quickInf.sharing.get().getSnapshot().devices.map((d) => d.matchCode)');
  log(`match codes on the host: ${hostCodes.join(', ')} (devices: ${windows.map((w) => w.code).join(', ')})`);

  // The host's share panel, as the person sees it.
  if (screenshot) {
    await until(host, '!!document.querySelector(".share-panel")', 'the share panel', 30).catch(() => log('the share panel isn’t showing'));
    const { data } = await host.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(screenshot, Buffer.from(data, 'base64'));
    log(`screenshot: ${screenshot}`);
  }

  let started = Date.now();
  const loaded = await evaluate(
    host,
    `(async () => window.__quickInf.client.load(${JSON.stringify(model)}, { share: await window.__quickInf.sharing.get().request() }).then((r) => r, (e) => ({ error: e.code + ': ' + e.message })))()`,
  );
  log(`loaded in ${((Date.now() - started) / 1000).toFixed(1)} s: ${JSON.stringify(loaded.shared ?? loaded.error)}`);
  for (const [i, w] of windows.entries()) log(`device ${i + 1}: ${JSON.stringify(await evaluate(w.cdp, 'window.__quickInfJoin.getSnapshot().peer'))}`);

  let answer = await reply(host, 'What is the capital of France?');
  log(`reply: ${JSON.stringify(answer)}`);

  // One device leaves: the next reply must fail as a disconnection, and loading again uses the rest.
  const leaving = windows.pop();
  await (await connect(first.webSocketDebuggerUrl)).send('Target.closeTarget', { targetId: leaving.targetId });
  await wait(2000);
  answer = await reply(host, 'Name a yellow fruit.');
  log(`after a device left: ${JSON.stringify(answer)}`);
  started = Date.now();
  const again = await evaluate(
    host,
    `(async () => window.__quickInf.client.load(${JSON.stringify(model)}, { share: await window.__quickInf.sharing.get().request() }).then((r) => r, (e) => ({ error: e.code + ': ' + e.message })))()`,
  );
  log(`loaded again in ${((Date.now() - started) / 1000).toFixed(1)} s: ${JSON.stringify(again.shared ?? again.error)}`);
  answer = await reply(host, 'Name a yellow fruit.');
  log(`reply: ${JSON.stringify(answer)}`);

  // The chat says it's shared, beside "Loaded".
  await evaluate(host, `[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'New chat')?.click()`);
  await until(host, '!!document.querySelector(".shared-badge")', 'the shared badge', 20).catch(() => log('no shared badge in the chat'));
  log(`chat header: ${await evaluate(host, 'document.querySelector(".chat__runtime")?.textContent ?? "(none)"')}`);
  if (screenshot) {
    const { data } = await host.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(screenshot.replace(/\.png$/, '-chat.png'), Buffer.from(data, 'base64'));
  }

  // The remaining device keeps its part (counted in its storage); when the host goes, it offers the way home.
  const device = windows[0];
  await until(device.cdp, 'Object.values(window.__quickInf.client.getStorage().models).some((m) => m.sharedPartBytes > 0)', 'the kept part in storage', 30).catch(() => {});
  log(`device storage: ${JSON.stringify(await evaluate(device.cdp, 'Object.fromEntries(Object.entries(window.__quickInf.client.getStorage().models).map(([id, m]) => [id, m.sharedPartBytes]))'))}`);
  host.close();
  await (await connect((await targets()).find((t) => t.id === device.targetId).webSocketDebuggerUrl)).send('Target.closeTarget', { targetId: first.id });
  await until(device.cdp, "window.__quickInfJoin.getSnapshot().phase === 'disconnected'", 'the device to notice the host left', 30).catch(() => {});
  log(`device after the host left: ${await evaluate(device.cdp, 'document.querySelector(".join-screen__card")?.innerText')}`);
  if (screenshot) {
    const { data } = await device.cdp.send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(screenshot.replace(/\.png$/, '-device.png'), Buffer.from(data, 'base64'));
  }
  }
} catch (err) {
  console.error(err);
  process.exitCode = 1;
} finally {
  chrome.kill();
  await wait(500);
  rmSync(profile, { recursive: true, force: true });
}

