#!/usr/bin/env node
// Drives the phone test suite on an Android phone over USB, so tests can be rerun and
// inspected from the Mac without touching the phone.
//
// Needs: adb (brew install --cask android-platform-tools), USB debugging on the phone,
// the dev server running (npm run dev), and Chrome open on the phone.
// `setup` forwards the dev server to the phone (https://localhost:5173 there) and Chrome's
// DevTools to the Mac (port 9222); the other commands talk to the suite's tab through it.
//
//   node tools/phone/phone.mjs setup
//   node tools/phone/phone.mjs run [options]   new run, e.g. "only=qwen3-0.6b&engine=webllm&quick&keep"
//   node tools/phone/phone.mjs clear           wipe everything the suite stored
//   node tools/phone/phone.mjs reload          reload the suite page (an interrupted test reruns)
//   node tools/phone/phone.mjs status [lines]  the end of the suite page's text (default 30 lines)
//   node tools/phone/phone.mjs watch           stream the page's output, console errors and exceptions
//   node tools/phone/phone.mjs eval <js>       run JavaScript in the suite page, print the result
//   node tools/phone/phone.mjs shot [file]     screenshot of the phone's page (PNG)
//   node tools/phone/phone.mjs storage         what the suite's origin has stored

import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';

const ORIGIN = 'https://localhost:5173';
const SUITE = `${ORIGIN}/phone-suite.html`;
const DEVTOOLS = 'http://127.0.0.1:9222';
const [command = 'status', ...args] = process.argv.slice(2);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function adb(...a) {
  return execFileSync('adb', a, { encoding: 'utf8' }).trim();
}

function setup() {
  const devices = adb('devices').split('\n').slice(1).filter((l) => l.trim());
  if (!devices.some((l) => /\tdevice$/.test(l))) {
    throw new Error(`No authorised phone over USB (${devices.join('; ') || 'none'}). Plug it in and allow USB debugging.`);
  }
  adb('reverse', 'tcp:5173', 'tcp:5173');
  adb('forward', 'tcp:9222', 'localabstract:chrome_devtools_remote');
}

/** Pages served by the Mac's dev server (the only tabs this tool ever touches). */
const isOurs = (t) => t.type === 'page' && (t.url.startsWith(ORIGIN) || /^https?:\/\/192\.168\.\d+\.\d+:(5173|8090)/.test(t.url));

/**
 * The suite's tab on the phone: exactly one, in front (Chrome pauses background tabs).
 * Extra copies of our pages (e.g. after Chrome restarts) are closed; other tabs are never touched.
 */
async function suiteTab() {
  const list = async () => (await (await fetch(`${DEVTOOLS}/json`)).json()).filter(isOurs);
  let tabs;
  try {
    tabs = await list();
  } catch {
    setup();
    tabs = await list();
  }
  let tab = tabs.find((t) => t.url.startsWith(SUITE));
  if (!tab) {
    adb('shell', 'am', 'start', '-a', 'android.intent.action.VIEW', '-d', SUITE, 'com.android.chrome');
    for (let i = 0; i < 20 && !tab; i++) {
      await wait(500);
      tab = (await list()).find((t) => t.url.startsWith(SUITE));
    }
    if (!tab) throw new Error('Could not open the suite in Chrome on the phone.');
  }
  for (const other of (await list()).filter((t) => t.id !== tab.id)) await fetch(`${DEVTOOLS}/json/close/${other.id}`);
  await fetch(`${DEVTOOLS}/json/activate/${tab.id}`);
  return tab;
}

/** A DevTools connection to the suite's tab. */
async function connect() {
  const tab = await suiteTab();
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error('DevTools connection failed (is another DevTools window attached to this tab?)'));
  });
  let id = 0;
  const pending = new Map();
  const listeners = new Set();
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (pending.has(m.id)) {
      pending.get(m.id)(m);
      pending.delete(m.id);
    } else for (const l of listeners) l(m);
  };
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const i = ++id;
      pending.set(i, (m) => (m.error ? reject(new Error(m.error.message)) : resolve(m.result)));
      ws.send(JSON.stringify({ id: i, method, params }));
    });
  const evaluate = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value;
  };
  return { tab, send, evaluate, on: (l) => listeners.add(l), close: () => ws.close() };
}

const pageText = `document.getElementById('out')?.textContent ?? document.body.innerText`;

async function navigate(url) {
  const c = await connect();
  await c.send('Page.navigate', { url });
  c.close();
  console.log(`opened ${url}`);
}

async function main() {
  switch (command) {
    case 'setup': {
      setup();
      console.log('Forwarded the dev server to the phone (https://localhost:5173) and its Chrome DevTools to port 9222.');
      console.log(`Model: ${adb('shell', 'getprop', 'ro.product.model')}`);
      return;
    }
    case 'run':
      return navigate(`${SUITE}?reset${args[0] ? `&${args[0].replace(/^\?/, '')}` : ''}`);
    case 'clear':
      return navigate(`${SUITE}?clear`);
    case 'reload': {
      const c = await connect();
      await c.send('Page.reload', {});
      c.close();
      console.log('reloaded');
      return;
    }
    case 'status': {
      const c = await connect();
      const text = await c.evaluate(pageText);
      c.close();
      console.log(text.split('\n').slice(-Number(args[0] ?? 30)).join('\n'));
      return;
    }
    case 'eval': {
      const c = await connect();
      const value = await c.evaluate(args.join(' '));
      c.close();
      console.log(typeof value === 'string' ? value : JSON.stringify(value, null, 2));
      return;
    }
    case 'storage': {
      const c = await connect();
      const value = await c.evaluate(`(async () => {
        const sizes = {};
        const walk = async (d, path) => { for await (const [n, h] of d.entries()) { if (h.kind === 'directory') await walk(h, [...path, n]); else { const k = path.slice(0, 2).join('/') || '(root)'; sizes[k] = (sizes[k] ?? 0) + (await h.getFile()).size / 1e6; } } };
        await walk(await navigator.storage.getDirectory(), []);
        const e = await navigator.storage.estimate();
        return { usageMB: Math.round(e.usage / 1e6), quotaMB: Math.round(e.quota / 1e6), files: Object.fromEntries(Object.entries(sizes).map(([k, v]) => [k, Math.round(v)])), databases: (await indexedDB.databases()).map((d) => d.name) };
      })()`);
      c.close();
      console.log(JSON.stringify(value, null, 2));
      return;
    }
    case 'shot': {
      const c = await connect();
      const { data } = await c.send('Page.captureScreenshot', { format: 'png' });
      c.close();
      const file = args[0] ?? 'phone.png';
      writeFileSync(file, Buffer.from(data, 'base64'));
      console.log(`saved ${file}`);
      return;
    }
    case 'watch': {
      const c = await connect();
      await c.send('Runtime.enable');
      await c.send('Log.enable');
      c.on((m) => {
        if (m.method === 'Runtime.exceptionThrown') console.log(`EXCEPTION ${m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text}`);
        else if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) {
          console.log(`${m.params.type.toUpperCase()} ${m.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 500)}`);
        } else if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') console.log(`LOG ${m.params.entry.text.slice(0, 500)}`);
      });
      // New lines of the page's own output, every two seconds (re-attaching after reloads).
      let seen = '';
      for (;;) {
        const text = await c.evaluate(pageText).catch(() => null);
        if (text !== null) {
          const fresh = text.startsWith(seen) ? text.slice(seen.length) : `\n[page reloaded]\n${text}`;
          if (fresh.trim()) process.stdout.write(fresh.endsWith('\n') ? fresh : `${fresh}\n`);
          seen = text;
        }
        await wait(2000);
      }
    }
    default:
      throw new Error(`Unknown command ${command}. See the top of tools/phone/phone.mjs.`);
  }
}

main().catch((err) => {
  console.error(err.message ?? err);
  process.exit(1);
});
