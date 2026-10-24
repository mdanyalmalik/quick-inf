// Development: one model split across several windows or devices (info/distributed-plan.md, milestones 2–3). One
// window hosts (the chat, the embedding, the output layer and the first layers); the others join its room and run
// the layers the host's plan gives them. Messages go over a BroadcastChannel between windows of this browser
// (optionally delayed like a network), or with `rtc` over WebRTC data channels, introduced through the relay
// (api/_relay.ts): the host shows a join link and QR code carrying the room's secret after #.
//
//   host: /pipeline.html?model=<catalogue id>&room=<name>[&mem=<GB>][&bw=<GB/s>][&delay=<ms>][&context=N]
//                       [&peers=N&autorun]   wait for N devices, then load and run the checks (tools/bench/pipeline.mjs)
//   peer: /pipeline.html?join=<room>[&mem=<GB>][&bw=<GB/s>][&label=<name>]
//   WebRTC: the host adds &rtc (no room); devices open its join link, /pipeline.html?join&rtc[&mem=…]#<secret>
//
// mem and bw are what the window claims to have (memory for its share, effective bandwidth), standing in for each
// device's benchmark, so the plan can be uneven. Model files come from the dev server's cache by byte range, so each
// window reads only its layers. Runs in a worker (binary range reads); the page shows the log and, on the host, a chat
// box. The host's results are exposed as `window.__pipeline`.

import { artifactUrl, findModel, type ModelSpec, type OnnxArtifact } from '../catalogue/models';
import { getGpuDevice } from '../device/gpu';
import type { GenerationOptions, LoadedArtifacts } from '../runtime/contract';
import { OnnxWebGpuAdapter, preparedLayerSplit } from '../runtime/onnx/onnx-adapter';
import { getOrt, loadOrt } from '../runtime/onnx/ort-env';
import type { ChatTurn } from '../tokenizer/prompt-format';
import { planSplit, type Plan } from '../distributed/planner';
import { broadcastChannel, buffersOf, type Channel, type DeviceInfo, type PipelineMessage } from '../distributed/protocol';
import { createRemoteChain, PipelineLink } from '../distributed/remote-stage';
import { startPeer } from '../distributed/peer-stage';
import { newSecret } from '../distributed/relay-client';
import { answerLink, hostRoom, joinRoom, offerLink } from '../distributed/webrtc';
import qrcode from 'qrcode-generator';
import { devFileStats, rangedFile } from './dev-files';
import { referenceFor, tokenizerFixturesFor } from './fixtures';

const inWorker = typeof document === 'undefined';
const params = new URLSearchParams(inWorker ? self.name : location.search);
const GB = 1e9;
const device: DeviceInfo = {
  label: params.get('label') ?? (params.has('join') ? 'peer' : 'host'),
  memoryBytes: Number(params.get('mem') ?? 4) * GB,
  bandwidthGBs: Number(params.get('bw') ?? 100),
};

type State = { done: boolean; error?: string; log: string[]; results: Record<string, unknown>; secret?: string; joinUrl?: string };
/** `pipe`: a pipeline message the page carries over WebRTC for the worker (workers have no WebRTC). */
type FromWorker = { log?: string; delta?: string; final?: Omit<State, 'log'>; pipe?: PipelineMessage };
type ToWorker = { prompt?: string; pipe?: PipelineMessage };
const rtc = params.has('rtc');
/** How long two devices get to link directly before their runs go through the host instead. */
const LINK_TIMEOUT_MS = 10_000;

/** In the worker: the pipeline's channel, through the page (WebRTC) or a BroadcastChannel. */
function workerChannel(room: string, delayMs: number): Channel {
  if (!rtc) return broadcastChannel(room, delayMs);
  return {
    send: (message) => self.postMessage({ pipe: message } satisfies FromWorker, { transfer: buffersOf(message) }),
    onMessage: (handler) => addEventListener('message', (event: MessageEvent<ToWorker>) => event.data.pipe && handler(event.data.pipe)),
    close() {},
  };
}

if (!inWorker) page();
else {
  const state: State = { done: false, log: [], results: {} };
  const log = (line: string) => self.postMessage({ log: line } satisfies FromWorker);
  const finish = () => self.postMessage({ final: { done: true, error: state.error, results: state.results } } satisfies FromWorker);
  (params.has('join') ? peer(log) : host(state, log))
    .catch((err: unknown) => {
      state.error = err instanceof Error ? `${err.name}: ${err.message}\n${err.stack ?? ''}` : String(err);
      log(`error: ${state.error}`);
      state.done = true;
      finish();
    });
  if (!params.has('join')) addEventListener('pipeline-finished', () => finish());
}

/** The page: the log, and the host's chat box. */
function page(): void {
  const state: State = { done: false, log: [], results: {} };
  (window as unknown as { __pipeline: State }).__pipeline = state;
  const out = document.getElementById('out')!;
  const reply = document.getElementById('reply')!;
  const form = document.getElementById('chat') as HTMLFormElement;
  const append = (line: string) => {
    state.log.push(line);
    out.textContent += line + '\n';
  };
  const room = params.get('room') ?? params.get('join');
  const role = document.getElementById('role')!;
  role.textContent = params.has('join')
    ? `Device "${device.label}"${rtc ? '' : ` in room ${room}`}`
    : `Host of room ${room}: open ${location.origin}/pipeline.html?join=${room}&mem=2 in other windows`;
  if (params.has('join')) form.hidden = true;
  // WebRTC: the host's channel to each device, by the pipeline id its worker uses (learned from the messages it
  // sends); a device's channels to the host and to the devices it's linked with.
  const devices = new Map<string, Channel>();
  let toHost: Channel | null = null;
  const neighbours = new Map<string, Channel>();
  let me = '';
  let worker: Worker | null = null;
  const toWorker = (pipe: PipelineMessage) => worker!.postMessage({ pipe } satisfies ToWorker, { transfer: buffersOf(pipe) });
  const startWorker = () => {
    worker = new Worker(new URL(import.meta.url), { type: 'module', name: location.search });
    worker.onmessage = (event: MessageEvent<FromWorker>) => {
      const { pipe } = event.data;
      if (event.data.log !== undefined) append(event.data.log);
      if (event.data.delta !== undefined) reply.textContent += event.data.delta;
      if (event.data.final) Object.assign(state, event.data.final);
      if (pipe && toHost) {
        // A device: to the device a run goes on to if it's linked, else to the host (which passes it on).
        if (pipe.type === 'hello') me = pipe.peer;
        if (pipe.type === 'run' && neighbours.has(pipe.peer)) neighbours.get(pipe.peer)!.send(pipe);
        else toHost.send(pipe);
      } else if (pipe) {
        if ('peer' in pipe && devices.has(pipe.peer)) devices.get(pipe.peer)!.send(pipe);
        else for (const channel of new Set(devices.values())) channel.send(pipe);
      }
    };
    worker.onerror = (event) => {
      state.error = `worker: ${event.message}`;
      state.done = true;
      append(state.error);
    };
  };
  if (!rtc) startWorker();
  else if (params.has('join')) {
    // A device: connect to the host through the room in the link, then run its layers.
    const secret = location.hash.slice(1);
    joinRoom(secret, device.label, append)
      .then((connected) => {
        append(`connected to the host directly; match code ${connected.matchCode}`);
        role.textContent = `Device "${device.label}": connected (match code ${connected.matchCode})`;
        const host = connected.channel;
        toHost = host;
        startWorker();
        const linked = (to: string, channel: Channel | null) => {
          if (channel) {
            neighbours.set(to, channel);
            channel.onMessage(toWorker);
          }
          append(channel ? `linked directly to ${to}` : `no direct link to ${to}: runs go through the host`);
          host.send({ type: 'linked', peer: me, to, direct: channel !== null });
        };
        const links = new Map<string, { accept(sdp: string): Promise<void> }>();
        host.onMessage((pipe) => {
          // Links to other devices are set up here (workers have no WebRTC), their offers and answers through the host.
          if (pipe.type === 'link') {
            void offerLink((sdp) => host.send({ type: 'signal', peer: pipe.to, from: me, kind: 'offer', sdp })).then(({ accept, open }) => {
              links.set(pipe.to, { accept });
              void Promise.race([open, new Promise<null>((r) => setTimeout(() => r(null), LINK_TIMEOUT_MS))]).then((channel) => linked(pipe.to, channel));
            });
          } else if (pipe.type === 'signal' && pipe.kind === 'offer') {
            void answerLink(pipe.sdp, (sdp) => host.send({ type: 'signal', peer: pipe.from, from: me, kind: 'answer', sdp })).then((channel) => {
              neighbours.set(pipe.from, channel);
              channel.onMessage(toWorker);
            });
          } else if (pipe.type === 'signal') void links.get(pipe.from)?.accept(pipe.sdp);
          else toWorker(pipe);
        });
      })
      .catch((err: unknown) => append(`could not join: ${String(err)}`));
  } else {
    // The host: a new room, its join link and QR code; each device's channel once it connects.
    const secret = newSecret();
    const joinUrl = `${location.origin}/pipeline.html?join&rtc#${secret}`;
    Object.assign(state, { secret, joinUrl });
    role.innerHTML = '';
    const qr = qrcode(0, 'M');
    qr.addData(joinUrl);
    qr.make();
    const figure = document.createElement('div');
    figure.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 4 });
    const link = document.createElement('a');
    link.href = joinUrl;
    link.textContent = joinUrl;
    role.append('Host: scan or open this on the other devices', figure, link);
    startWorker();
    hostRoom(
      secret,
      (connected) => {
        append(`${connected.label} connected directly; match code ${connected.matchCode}`);
        connected.channel.onMessage((pipe) => {
          // Signals and runs a device sends on are for another device: passed straight on.
          if (pipe.type === 'signal' || pipe.type === 'run') {
            if (pipe.type === 'signal') devices.set(pipe.from, connected.channel);
            devices.get(pipe.peer)?.send(pipe);
            return;
          }
          if ('peer' in pipe) devices.set(pipe.peer, connected.channel);
          toWorker(pipe);
        });
      },
      append,
    ).catch((err: unknown) => append(`could not open the room: ${String(err)}`));
  }
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const input = form.elements.namedItem('prompt') as HTMLInputElement;
    reply.textContent += `\n\n> ${input.value}\n`;
    worker?.postMessage({ prompt: input.value } satisfies ToWorker);
    input.value = '';
  });
}

async function files(model: ModelSpec, artifact: OnnxArtifact): Promise<LoadedArtifacts> {
  const byName = (list: OnnxArtifact['externalData']) =>
    Object.fromEntries((list ?? []).map((f) => [f.path.slice(f.path.lastIndexOf('/') + 1), rangedFile(model, f)] as const));
  const tokenizer = await (await fetch(artifactUrl(model, model.tokenizer))).arrayBuffer();
  return {
    graph: rangedFile(model, artifact.graph),
    externalData: byName(artifact.externalData),
    tokenizerJson: JSON.parse(new TextDecoder().decode(tokenizer)) as unknown,
  };
}

async function peer(log: (line: string) => void): Promise<void> {
  await loadOrt();
  const gpu = await getGpuDevice();
  const id = `${device.label}-${Math.random().toString(36).slice(2, 7)}`;
  log(`${id}: ${(device.memoryBytes / GB).toFixed(1)} GB, ${device.bandwidthGBs} GB/s; waiting for the host`);
  startPeer({
    id,
    device,
    gpu,
    channel: workerChannel(params.get('join')!, Number(params.get('delay') ?? 0)),
    files: async (artifact, modelId) => files(findModel(modelId)!, artifact),
    log: (line) => log(`${id}: ${line}`),
    loadDetail: () => `${(devFileStats.bytesRead / 1e9).toFixed(2)} GB fetched`,
  });
}

async function host(state: State, log: (line: string) => void): Promise<void> {
  const model = findModel(params.get('model') ?? 'llama-3.2-3b-instruct');
  if (!model) throw new Error('Unknown model');
  const artifact = model.artifacts.find((a): a is OnnxArtifact => a.engine === 'onnx-webgpu');
  if (!artifact) throw new Error('No ONNX artifact');
  const delayMs = Number(params.get('delay') ?? 0);
  const contextTokens = Number(params.get('context') ?? model.generation.contextTokens);
  const channel = workerChannel(params.get('room') ?? 'test', delayMs);
  const link = new PipelineLink(channel);
  const peers = new Map<string, DeviceInfo>();
  channel.onMessage((m) => {
    if (m.type === 'hello' && !peers.has(m.peer)) {
      peers.set(m.peer, m.device);
      log(`joined: ${m.peer} (${(m.device.memoryBytes / GB).toFixed(1)} GB, ${m.device.bandwidthGBs} GB/s)`);
    } else if (m.type === 'bye') peers.delete(m.peer);
  });
  channel.send({ type: 'host' });
  log(`hosting ${model.name}: ${(device.memoryBytes / GB).toFixed(1)} GB, ${device.bandwidthGBs} GB/s, ${delayMs} ms per hop`);

  const wanted = Number(params.get('peers') ?? 0);
  if (params.has('autorun')) {
    for (const start = performance.now(); peers.size < wanted; ) {
      if (performance.now() - start > 120_000) throw new Error(`Only ${peers.size} of ${wanted} devices joined`);
      await new Promise((r) => setTimeout(r, 250));
    }
  } else {
    log('send any message to plan and load with the devices that have joined');
    await nextPrompt();
  }

  // The plan, from each layer's weights: the prepared graph cut before every layer (and before the output layer).
  // Weights several layers use (rotary tables) count once per device; the host also holds the embedding (with its
  // first layer) and the output layer, each in a session of its own.
  const loaded = await files(model, artifact);
  const layers = model.architecture.layers;
  const perLayer = preparedLayerSplit(artifact, loaded, Array.from({ length: layers }, (_, i) => i + 1)).segments;
  const users = new Map<string, number>();
  for (const s of perLayer.slice(1, layers)) for (const name of s.initializers.keys()) users.set(name, (users.get(name) ?? 0) + 1);
  const shared = new Set([...users].filter(([, n]) => n > 1).map(([name]) => name));
  const own = (s: (typeof perLayer)[number]) => [...s.initializers].reduce((n, [name, bytes]) => n + (shared.has(name) ? 0 : bytes), 0);
  const sharedBytes = [...perLayer[1]!.initializers].reduce((n, [name, bytes]) => n + (shared.has(name) ? bytes : 0), 0);
  const layerBytes = perLayer.slice(0, layers).map(own);
  const typical = [...layerBytes].sort((a, b) => a - b)[Math.floor(layers / 2)]!;
  const hostBytes = layerBytes[0]! - typical + own(perLayer[layers]!);
  layerBytes[0] = typical;
  const { kvHeads, headDim } = model.architecture;
  const kvBytesPerLayer = 2 * kvHeads * headDim * 2 * contextTokens;
  log(
    `layers: ${(layerBytes.reduce((a, b) => a + b, 0) / GB).toFixed(2)} GB of weights over ${layers}, ${(sharedBytes / 1e6).toFixed(0)} MB shared by them, ` +
      `host's extra ${(hostBytes / GB).toFixed(2)} GB, KV ${(kvBytesPerLayer / 1e6).toFixed(0)} MB per layer at ${contextTokens} tokens`,
  );
  const plan = planSplit({
    layerBytes,
    hostBytes,
    kvBytesPerLayer,
    overheadBytes: 0.25 * GB + sharedBytes,
    devices: [{ id: 'host', ...device }, ...[...peers].map(([id, d]) => ({ id, ...d }))],
    hopMs: delayMs,
  });
  if (!plan) throw new Error('The devices that joined cannot hold the model together');
  log(`plan: ${describe(plan)}; about ${plan.msPerToken.toFixed(1)} ms per token`);
  state.results.plan = plan;

  // Segments: the host's layers, each device's, then the output layer back on the host.
  const cuts = [...plan.cuts, layers];
  const owner = (index: number) => (index === 0 || index === cuts.length ? null : plan.shares[index]!.id);
  const adapter = new OnnxWebGpuAdapter();
  const formatOptions = { date: (tokenizerFixturesFor(model.id)?.templateKwargs as { date_string?: string } | undefined)?.date_string };
  const start = performance.now();
  await adapter.load(model, artifact, loaded, () => {}, {
    formatOptions,
    splitLayers: cuts,
    // The devices' segments (all between the host's two) as one chain: each passes a step's values to the next.
    remoteStage: (split, index, types) => {
      const peers: string[] = [];
      for (let i = index; owner(i); i++) peers.push(owner(i)!);
      return peers.length ? createRemoteChain({ link, peers, model, artifactId: artifact.id, cuts, split, index, types, Tensor: getOrt().Tensor }) : null;
    },
  });
  log(`loaded in ${((performance.now() - start) / 1000).toFixed(1)} s`);
  state.results.loadMs = Math.round(performance.now() - start);

  if (params.has('autorun')) {
    state.results.checks = await checks(adapter, model, contextTokens, log);
    state.done = true;
    dispatchEvent(new Event('pipeline-finished'));
  }
  const turns: ChatTurn[] = [];
  for (;;) {
    const prompt = await nextPrompt();
    turns.push({ role: 'user', content: prompt });
    const res = await adapter.generate({ turns, options: { ...model.generation, maxNewTokens: 512, contextTokens } }, (text) =>
      self.postMessage({ delta: text } satisfies FromWorker),
    );
    turns.push({ role: 'assistant', content: res.text });
    log(`${res.tokenIds.length} tokens, ${res.stats.decodeTokensPerSecond?.toFixed(1)} tok/s`);
  }
}

function nextPrompt(): Promise<string> {
  return new Promise((resolve) => {
    const listener = (event: MessageEvent<ToWorker>) => {
      if (event.data.prompt === undefined) return;
      removeEventListener('message', listener);
      resolve(event.data.prompt);
    };
    addEventListener('message', listener);
  });
}

function describe(plan: Plan): string {
  return plan.shares.map((s) => `${s.id} layers ${s.from}–${s.to - 1} (${(s.bytes / GB).toFixed(2)} GB)`).join(', ');
}

/** The reference fixture's greedy cases, a follow-up turn on the kept cache, and speed. */
async function checks(adapter: OnnxWebGpuAdapter, model: ModelSpec, contextTokens: number, log: (line: string) => void): Promise<unknown[]> {
  const reference = referenceFor(model.id, 'q4f16');
  const greedy = (maxNewTokens: number): GenerationOptions => ({ maxNewTokens, temperature: 0, topP: 1, repetitionPenalty: 1, repetitionWindow: 64, contextTokens });
  const results: unknown[] = [];
  for (const [i, c] of (reference?.cases ?? []).entries()) {
    const expected = c.greedyIds;
    const res = await adapter.generate({ turns: c.messages as ChatTurn[], options: greedy(expected.length) }, () => {});
    let firstDiff = -1;
    for (let k = 0; k < res.tokenIds.length; k++) {
      if (res.tokenIds[k] !== expected[k]) {
        firstDiff = k;
        break;
      }
    }
    const top = firstDiff === -1 ? undefined : c.topK[firstDiff];
    const nearTie = top !== undefined && top.logits[0]! - top.logits[1]! < 0.25;
    results.push({ name: `Greedy #${i + 1}`, pass: firstDiff === -1 || nearTie, firstDiff, text: res.text.slice(0, 80) });
    log(`greedy #${i + 1}: ${firstDiff === -1 ? 'identical' : `differs at ${firstDiff}${nearTie ? ' (near tie)' : ''}`}`);
  }
  // A second turn continues the first one's cache on every device.
  const turns: ChatTurn[] = [{ role: 'user', content: 'Name a fruit that is yellow.' }];
  const first = await adapter.generate({ turns, options: greedy(24) }, () => {});
  turns.push({ role: 'assistant', content: first.text }, { role: 'user', content: 'And one that is red?' });
  const second = await adapter.generate({ turns, options: greedy(24) }, () => {});
  results.push({ name: 'Follow-up turn', pass: second.stats.reusedPromptTokens > 0 && second.text.length > 0, reused: second.stats.reusedPromptTokens, text: second.text });
  log(`follow-up: reused ${second.stats.reusedPromptTokens} tokens: ${second.text.slice(0, 60)}`);
  // Speed.
  const perf = await adapter.generate({ turns: [{ role: 'user', content: 'Explain how a refrigerator works in detail.' }], options: greedy(128) }, () => {});
  results.push({ name: 'Speed', decodeTokensPerSecond: perf.stats.decodeTokensPerSecond, firstTokenMs: perf.stats.timeToFirstTokenMs });
  log(`speed: ${perf.stats.decodeTokensPerSecond?.toFixed(1)} tok/s, first token ${Math.round(perf.stats.timeToFirstTokenMs ?? 0)} ms`);
  const segments = (globalThis as { __quickInfSegments?: { stats: unknown } }).__quickInfSegments;
  if (segments) results.push({ name: 'Segments', stats: segments.stats });
  return results;
}
