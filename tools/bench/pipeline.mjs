#!/usr/bin/env node
// Runs pipeline.html (one model split across windows: info/distributed-plan.md) in Chrome: a window per device, then
// the host, which waits for them, plans, loads and runs its checks. Prints the host's results and every window's log.
// Needs the dev server (npx vite --port 5174). Uses a temporary Chrome profile, deleted afterwards.
//
//   node tools/bench/pipeline.mjs [--headed] [--delay MS] [--rtc] [--host QUERY] model peer-query ...
//
// --rtc connects the windows over WebRTC through the relay (the host first, then each device opens its join link).
//   e.g. node tools/bench/pipeline.mjs --host 'mem=1.6' llama-3.2-3b-instruct 'mem=1&bw=100' 'mem=0.8&bw=60'

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const ORIGIN = 'https://localhost:5174';
const args = process.argv.slice(2);
let headed = false;
let delay = 0;
let rtc = false;
let hostQuery = '';
const rest = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--headed') headed = true;
  else if (args[i] === '--delay') delay = Number(args[++i]);
  else if (args[i] === '--rtc') rtc = true;
  else if (args[i] === '--host') hostQuery = args[++i];
  else rest.push(args[i]);
}
const [model, ...peerQueries] = rest;
if (!model) {
  console.error('Usage: node tools/bench/pipeline.mjs [--headed] [--delay MS] [--host QUERY] model peer-query ...');
  process.exit(1);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), 'qi-pipeline-'));
// Stopped from outside (guarded.sh when swap grows, or Ctrl-C): the browser and its profile (a copy of the model,
// often gigabytes) go too, as `finally` doesn't run then.
const stopped = (signal) => {
  try {
    chrome.kill('SIGKILL');
  } catch {
    // not started yet
  }
  rmSync(profile, { recursive: true, force: true });
  process.exit(signal === 'SIGINT' ? 130 : 143);
};
process.on('SIGTERM', stopped);
process.on('SIGINT', stopped);
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

const evaluate = async (cdp, expression) => (await cdp.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result.value;

try {
  const room = `r${Date.now().toString(36)}`;
  const first = (await targets()).find((t) => t.type === 'page');
  const browser = await connect(first.webSocketDebuggerUrl);
  const hostUrl = `${ORIGIN}/pipeline.html?model=${encodeURIComponent(model)}&${rtc ? 'rtc' : `room=${room}&delay=${delay}`}&peers=${peerQueries.length}&autorun${hostQuery ? '&' + hostQuery : ''}`;
  // Each window's script must be running (its first load can stall while the dev server bundles): load it once
  // more if it isn't after a while.
  const started = async (cdp, url, check) => {
    for (let attempt = 0; attempt < 2; attempt++) {
      if (attempt > 0) await cdp.send('Page.navigate', { url });
      for (let t = 0; t < 60; t++) {
        await wait(1000);
        if (await evaluate(cdp, check).catch(() => false)) return true;
      }
    }
    return false;
  };
  // Over WebRTC the host comes first: its room's secret goes in each device's join link.
  let secret = '';
  if (rtc) {
    await browser.send('Page.navigate', { url: hostUrl });
    if (!(await started(browser, hostUrl, '!!(window.__pipeline && window.__pipeline.secret)'))) throw new Error("the host's window did not start");
    secret = await evaluate(browser, 'window.__pipeline.secret');
  }
  // A window per device, each its own target (separate windows, so none is a hidden tab).
  const peers = [];
  for (const [i, query] of peerQueries.entries()) {
    const url = rtc
      ? `${ORIGIN}/pipeline.html?join&rtc&label=device${i + 1}&${query}#${secret}`
      : `${ORIGIN}/pipeline.html?join=${room}&label=device${i + 1}&delay=${delay}&${query}`;
    const { targetId } = await browser.send('Target.createTarget', { url, newWindow: true });
    peers.push(targetId);
    const target = (await targets()).find((t) => t.id === targetId);
    const cdp = await connect(target.webSocketDebuggerUrl);
    const ready = await started(cdp, url, 'window.__pipeline ? window.__pipeline.log.length > 0 : false');
    cdp.close();
    if (!ready) throw new Error(`device${i + 1}'s window did not start`);
  }
  if (!rtc) await browser.send('Page.navigate', { url: hostUrl });
  const startedAt = Date.now();
  let result = null;
  for (;;) {
    await wait(1000);
    result = await evaluate(browser, 'window.__pipeline && window.__pipeline.done ? JSON.stringify(window.__pipeline) : null').catch(() => null);
    if (result) break;
    if (Date.now() - startedAt > 20 * 60 * 1000) {
      result = JSON.stringify({ error: 'timed out', log: await evaluate(browser, 'window.__pipeline && window.__pipeline.log').catch(() => null) });
      break;
    }
  }
  const host = JSON.parse(result);
  const all = await targets();
  for (const [i, targetId] of peers.entries()) {
    const target = all.find((t) => t.id === targetId);
    if (!target) continue;
    const cdp = await connect(target.webSocketDebuggerUrl);
    const log = await evaluate(cdp, 'window.__pipeline && window.__pipeline.log').catch(() => null);
    console.log(`--- device${i + 1} log\n${(log ?? []).join('\n')}`);
    cdp.close();
  }
  console.log(`--- host log\n${(host.log ?? []).join('\n')}`);
  console.log(JSON.stringify({ model, s: Math.round((Date.now() - startedAt) / 1000), error: host.error, ...host.results }));
  browser.close();
} finally {
  chrome.kill();
  await wait(500);
  rmSync(profile, { recursive: true, force: true });
}
