// Installs our own ONNX Runtime Web build over the matching files of the onnxruntime-web package (postinstall, after
// tools/patch-ort.mjs). The build is ONNX Runtime with a few patches of ours (faster kernels), published with its
// patch and build script at github.com/<repo> (see release.json); its WebGPU binaries and their glue replace
// ort-wasm-simd-threaded.{asyncify,jspi}.{wasm,mjs}, which ort-env.ts loads as pairs.
//
// Files are checked against release.json's sizes and SHA-256 and kept in a cache (~/.cache/quick-inf/onnxruntime),
// so a reinstall works offline. Fails loudly on a download or checksum problem, or another onnxruntime-web version.
// QI_STOCK_ORT=1 keeps npm's files (to compare against them).

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const manifest = JSON.parse(readFileSync(join(root, 'tools/ort/release.json'), 'utf8'));
const dist = join(root, 'node_modules/onnxruntime-web/dist');
const fail = (message) => {
  console.error(`tools/ort/install.mjs: ${message}`);
  process.exit(1);
};

if (process.env.QI_STOCK_ORT === '1') {
  console.log('tools/ort/install.mjs: QI_STOCK_ORT=1, keeping npm’s ONNX Runtime binaries');
  process.exit(0);
}
const installed = JSON.parse(readFileSync(join(root, 'node_modules/onnxruntime-web/package.json'), 'utf8')).version;
if (installed !== manifest.onnxruntimeWeb) {
  fail(`release ${manifest.tag} is built from onnxruntime-web ${manifest.onnxruntimeWeb}, but ${installed} is installed`);
}
const cache = join(process.env.XDG_CACHE_HOME ?? join(homedir(), '.cache'), 'quick-inf', 'onnxruntime', manifest.tag);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const matches = (bytes, file) => bytes.length === file.size && sha256(bytes) === file.sha256;

let replaced = 0;
for (const [name, file] of Object.entries(manifest.files)) {
  const target = join(dist, name);
  if (existsSync(target) && matches(readFileSync(target), file)) continue;
  const cached = join(cache, name);
  let bytes = existsSync(cached) ? readFileSync(cached) : null;
  if (!bytes || !matches(bytes, file)) {
    const url = `https://github.com/${manifest.repo}/releases/download/${manifest.tag}/${name}`;
    let response;
    try {
      response = await fetch(url);
    } catch (err) {
      fail(`couldn't download ${url}: ${err.message}`);
    }
    if (!response.ok) fail(`couldn't download ${url}: HTTP ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
    if (!matches(bytes, file)) fail(`${name} from ${url} doesn't match its size and SHA-256 in tools/ort/release.json`);
    mkdirSync(cache, { recursive: true });
    writeFileSync(cached, bytes);
  }
  writeFileSync(target, bytes);
  replaced++;
}
console.log(`tools/ort/install.mjs: ONNX Runtime Web ${manifest.tag} (${replaced ? `${replaced} files installed` : 'already installed'})`);
