// Development speech bench: loads a speech model through the app's own runtime client and worker (files from the
// dev server's local model cache), speaks the test sentences and reports per sentence the time to the first audio,
// the time to make it all and the real-time factor (making time over audio length), with each clip as a base64 WAV
// for scoring (tools/export/.cache/tts-eval). Driven by tools/bench/bench.mjs --page speech-bench.
//
//   /speech-bench.html?model=<speech model id>[&voice=<id>][&steps=N][&cpu=1]
// Results are exposed as `window.__bench`.

import { findSpeechModel } from '../catalogue/speech-models';
import { getRuntimeClient } from '../runtime/client';
import { encodeWav } from '../ui/workspaces/speech/wav';

const params = new URLSearchParams(location.search);
const modelId = params.get('model') ?? 'supertonic-3';
/** The evaluation sentences (tools/export/.cache/tts-eval/tts_gen.py SENTENCES). */
const SENTENCES = [
  'Hello from Phonon. This is a quick test.',
  'The quick brown fox jumps over the lazy dog near the riverbank.',
  'Please call me back at 555 0142 before 6 pm on Friday.',
  'Our meeting moved to March 14th, 2027, in room 302.',
  'Photosynthesis turns sunlight, water and carbon dioxide into sugar and oxygen.',
  'I would like two coffees and a blueberry muffin, please.',
  'Send the report to anna.lee@example.com by the end of the week.',
  'After the storm passed, the streets were quiet and the air smelled of rain.',
];

const state: { done: boolean; error?: string; log: string[]; results: Record<string, unknown> } = { done: false, log: [], results: {} };
(window as unknown as { __bench: typeof state }).__bench = state;
const out = document.getElementById('out')!;
const log = (line: string) => {
  state.log.push(line);
  out.textContent += line + '\n';
};

async function base64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let text = '';
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

async function run(): Promise<void> {
  const spec = findSpeechModel(modelId);
  if (!spec) throw new Error(`Unknown speech model ${modelId}`);
  const client = getRuntimeClient();
  const loadStart = performance.now();
  const loaded = await client.load(modelId);
  state.results.load = { ms: Math.round(performance.now() - loadStart), processor: loaded.processor, timings: loaded.timings };
  log(`loaded on the ${loaded.processor} in ${Math.round(performance.now() - loadStart)} ms`);
  const settings = {
    ...spec.defaults,
    ...(params.has('voice') && { voice: params.get('voice')! }),
    ...(params.has('steps') && { steps: Number(params.get('steps')) }),
  };
  const clips: unknown[] = [];
  for (const [i, text] of SENTENCES.entries()) {
    const pieces: Float32Array[] = [];
    const handle = client.generateSpeech({ text, settings }, (p) => pieces.push(p.audio));
    const r = await handle.result;
    const seconds = r.samples / r.sampleRate;
    clips.push({
      text,
      firstAudioMs: Math.round(r.timings.firstAudioMs),
      totalMs: Math.round(r.timings.totalMs),
      audioS: +seconds.toFixed(2),
      rtf: +(r.timings.totalMs / 1000 / seconds).toFixed(3),
      wav: await base64(encodeWav(pieces, r.sampleRate)),
    });
    log(`${i}: ${seconds.toFixed(2)} s of audio in ${Math.round(r.timings.totalMs)} ms (first audio ${Math.round(r.timings.firstAudioMs)} ms)`);
  }
  state.results.clips = clips;
}

run()
  .catch((err: unknown) => {
    state.error = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    log(`error: ${state.error}`);
  })
  .finally(() => {
    state.done = true;
  });
