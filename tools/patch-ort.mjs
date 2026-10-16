#!/usr/bin/env node
// Patches the installed ONNX Runtime Web (run by `npm install`, as postinstall) so the WebGPU execution provider's
// options can carry any of its native options: { name: 'webgpu', qiOptions: { enableGraphCapture: '1', ... } } passes
// each entry through. ONNX Runtime Web passes only a few by name, and passes enableGraphCapture only from the session
// option of that name, which also makes every run of the session reuse the first run's inputs and outputs: a session
// that runs prompts (other shapes, uncaptured) besides captured decoding steps can't use that (see
// src/runtime/onnx/onnx-adapter.ts: static caches). Others it lacks: enableInt64, maxNumPendingDispatches.
//
// Idempotent. Fails if a bundle doesn't have the expected code (another ONNX Runtime Web version): check the patch
// still applies, and update it, when upgrading.

import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dist = join(dirname(fileURLToPath(import.meta.url)), '..', 'node_modules', 'onnxruntime-web', 'dist');
const MARK = '/*qi-ep-options*/';
// Minified: `let{enableGraphCapture:X}=<session options>;if(typeof X=="boolean"&&X&&<append>(<options>,"enableGraphCapture","1",<allocs>),typeof <ep>.preferredLayout`.
const MINIFIED = /(let\{enableGraphCapture:(\w+)\}=\w+;)(if\(typeof \2=="boolean"&&\2&&(\w+)\((\w+),"enableGraphCapture","1",(\w+)\),typeof (\w+)\.preferredLayout)/g;
// Readable: `const { enableGraphCapture } = sessionOptions;` then `if (typeof enableGraphCapture === "boolean" && enableGraphCapture) {`.
const READABLE = /(const \{ enableGraphCapture \} = sessionOptions;)(\s*)(if \(typeof enableGraphCapture === "boolean" && enableGraphCapture\) \{)/g;

let patched = 0;
const failed = [];
for (const file of readdirSync(dist).filter((f) => /^ort\..*\.m?js$/.test(f))) {
  const path = join(dist, file);
  const code = readFileSync(path, 'utf8');
  if (!code.includes('"enableGraphCapture","1"') && !code.includes('"enableGraphCapture", "1"')) continue; // no WebGPU EP in this build
  if (code.includes(MARK)) continue;
  let count = 0;
  let next = code.replace(MINIFIED, (_m, head, _x, rest, append, options, allocs, ep) => {
    count++;
    return `${head}${MARK}if(${ep}.qiOptions)for(let[k,v]of Object.entries(${ep}.qiOptions))${append}(${options},k,String(v),${allocs});${rest}`;
  });
  next = next.replace(READABLE, (_m, head, space, rest) => {
    count++;
    return `${head}${MARK}${space}if (webgpuOptions.qiOptions) for (const [k, v] of Object.entries(webgpuOptions.qiOptions)) appendEpOption(epOptions, k, String(v), allocs);${space}${rest}`;
  });
  if (count === 0) {
    failed.push(file);
    continue;
  }
  writeFileSync(path, next);
  patched++;
}
if (failed.length) {
  console.error(`patch-ort: unexpected code in ${failed.join(', ')} (another onnxruntime-web version?); see tools/patch-ort.mjs`);
  process.exit(1);
}
if (patched) console.log(`patch-ort: patched ${patched} onnxruntime-web bundle(s): WebGPU EP options pass through`);
