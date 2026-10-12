// Image generation in the browser: each model's ONNX graphs as ONNX Runtime WebGPU sessions on the app's GPU
// device, run as its pipeline says (src/catalogue/image-models.ts). Runs in the worker.
//
// - 'sd' (SD Turbo, Stable Diffusion 1.5, LCM Dreamshaper): CLIP tokens → text encoder → UNet steps → VAE decoder.
//   Euler (SD Turbo re-noises between steps; SD 1.5 mixes prompt and negative prompt, "classifier-free guidance")
//   or LCM (guidance as a model input, a consistency step).
// - 'sdxl' (SDXL Turbo): two CLIP text encoders, whose outputs are joined, plus the second's pooled output and the
//   image size as extra UNet inputs; float16 throughout.
// - 'z-image' (Z-Image Turbo): the prompt in Qwen3's chat format → Qwen3 text encoder → a diffusion transformer
//   over flow matching → VAE decoder (16 latent channels).
// Latents are the model's [1, C, h/8, w/8] working image. Each step's predicted clean latents are decoded as a
// preview of the image so far (by a tiny decoder where there is one), the last step's as the image itself.

import type { InferenceSession, Tensor } from 'onnxruntime-web/webgpu';
import type { ArtifactFile } from '../../catalogue/models';
import {
  imageGraphs,
  outputSize,
  type ClipTokenizerFiles,
  type ImageModelSpec,
  type ImagePipeline,
  type ImageSettings,
} from '../../catalogue/image-models';
import { getGpuDevice } from '../../device/gpu';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { ClipTokenizer } from '../../tokenizer/clip-tokenizer';
import { RuntimeError } from '../contract';
import type { ByteSource } from '../onnx/external-weights';
import { createGraphSession } from '../onnx/onnx-adapter';
import { getOrt } from '../onnx/ort-env';
import { janusPromptIds, zImagePromptIds } from './prompts';
import {
  alphasCumprod,
  eulerSchedule,
  flowMatchSchedule,
  fromHalf,
  gaussian,
  guidanceEmbedding,
  lcmScalings,
  lcmTimesteps,
  toHalf,
  toRgba,
} from './schedulers';

export interface ImageGenerationRequest {
  prompt: string;
  settings: ImageSettings;
}

export interface ImageGenerationProgress {
  phase: 'prompt' | 'denoising' | 'decoding';
  /** Denoising steps done, of `steps`. */
  step: number;
  steps: number;
  /** The image as predicted after this step (several-step runs only). */
  preview?: Blob;
}

export interface ImageGenerationResult {
  png: Blob;
  width: number;
  height: number;
  seed: number;
  timings: { promptMs: number; denoiseMs: number; decodeMs: number; totalMs: number };
}

/** A model's files: the large ones to read on demand, small ones (tokenizers) whole. */
export interface ImageFileSource {
  source(file: ArtifactFile): ByteSource;
  bytes(file: ArtifactFile): Uint8Array<ArrayBuffer>;
}

type FloatType = 'float32' | 'float16';

/** One denoising step's prediction: the clean latents, and the latents to run next (after the last: unused). */
interface StepResult {
  clean: Float32Array;
  next: Float32Array;
}

interface Loaded {
  model: ImageModelSpec;
  pipeline: ImagePipeline;
  sessions: Map<string, InferenceSession>;
  clip?: ClipTokenizer;
  clip2?: ClipTokenizer;
  qwen?: BpeTokenizer;
}

const SD_LATENT_SCALE = 0.18215;
const SDXL_LATENT_SCALE = 0.13025;
const Z_IMAGE_LATENT = { scale: 0.3611, shift: 0.1159 };
/** Janus's image is a 24 × 24 grid of tokens. */
const JANUS_GRID = 24;
const JANUS_PREVIEW_ROWS = 4;

export class ImageEngine {
  private loaded: Loaded | null = null;
  private cancelRequested = false;

  get modelId(): string | null {
    return this.loaded?.model.id ?? null;
  }

  async load(model: ImageModelSpec, files: ImageFileSource): Promise<{ sessionMs: number }> {
    await this.unload();
    const pipeline = model.pipeline!;
    const start = performance.now();
    const loaded: Loaded = { model, pipeline, sessions: new Map(), ...tokenizers(pipeline, files) };
    let device: GPUDevice;
    try {
      device = await getGpuDevice();
    } catch (err) {
      throw new RuntimeError('load-failed', `The GPU could not be opened: ${message(err)}`, true);
    }
    try {
      // Smallest first, so a failure shows before the large graphs are read.
      const graphs = Object.entries(imageGraphs(pipeline)).sort(([, a], [, b]) => bytesOf(a) - bytesOf(b));
      for (const [role, g] of graphs) {
        const data = Object.fromEntries((g.data ?? []).map((f) => [fileName(f), files.source(f)]));
        // Janus's KV cache stays on the GPU between tokens.
        const gpuOutputs = pipeline.kind === 'janus' && role === 'languageModel' ? kvNames(pipeline.layers, 'present') : [];
        loaded.sessions.set(role, await createGraphSession(files.source(g.graph), device, data, gpuOutputs));
      }
      this.loaded = loaded;
    } catch (err) {
      await release(loaded);
      if (OUT_OF_MEMORY.test(message(err))) {
        throw new RuntimeError(
          'out-of-memory',
          `There isn’t enough free GPU memory to run ${model.name} on this device (${message(err)})`,
          true,
        );
      }
      throw new RuntimeError('load-failed', `The model could not be loaded on the GPU: ${message(err)}`, true);
    }
    return { sessionMs: performance.now() - start };
  }

  cancel(): void {
    this.cancelRequested = true;
  }

  async unload(): Promise<void> {
    const loaded = this.loaded;
    this.loaded = null;
    if (loaded) await release(loaded);
  }

  async generate(request: ImageGenerationRequest, onProgress: (p: ImageGenerationProgress) => void): Promise<ImageGenerationResult> {
    const loaded = this.loaded;
    if (!loaded) throw new RuntimeError('not-loaded', 'Load the model first.');
    this.cancelRequested = false;
    const { settings } = request;
    const start = performance.now();
    const chosen = Number(settings.seed ?? -1);
    const seed = chosen >= 0 ? Math.floor(chosen) : Math.floor(Math.random() * 2 ** 31);
    const { width, height } = outputSize(loaded.model, settings);
    const cancelled = () => this.cancelRequested;
    const run: ImageRun =
      loaded.pipeline.kind === 'janus'
        ? new JanusRun(loaded, request, seed, cancelled)
        : new DiffusionRun(
            loaded,
            request,
            Math.max(1, Math.min(50, Math.round(Number(settings.steps ?? 1)))),
            seed,
            width,
            height,
            cancelled,
          );
    const steps = run.steps;

    try {
      onProgress({ phase: 'prompt', step: 0, steps });
      await run.encodePrompt();
      const promptMs = performance.now() - start;

      const denoiseStart = performance.now();
      for (let i = 0; i < steps; i++) {
        if (this.cancelRequested) throw new RuntimeError('cancelled', 'Generation was stopped.');
        await run.step(i);
        const preview = i < steps - 1 ? await run.preview(i) : undefined;
        onProgress({ phase: 'denoising', step: i + 1, steps, preview });
      }
      run.dispose();
      const denoiseMs = performance.now() - denoiseStart;

      onProgress({ phase: 'decoding', step: steps, steps });
      const decodeStart = performance.now();
      const png = await run.final();
      const decodeMs = performance.now() - decodeStart;
      return { png, width, height, seed, timings: { promptMs, denoiseMs, decodeMs, totalMs: performance.now() - start } };
    } catch (err) {
      run.dispose();
      if (err instanceof RuntimeError) throw err;
      if (OUT_OF_MEMORY.test(message(err))) {
        throw new RuntimeError('out-of-memory', `There isn’t enough free GPU memory to run ${loaded.model.name} on this device`, true);
      }
      throw new RuntimeError('generation-failed', `Image generation failed: ${message(err)}`, true);
    }
  }
}

/** One generation, as the engine drives it: encode the prompt, take the steps (previewing between), finish. */
interface ImageRun {
  readonly steps: number;
  encodePrompt(): Promise<void>;
  step(i: number): Promise<void>;
  /** The image so far, after step `i`. */
  preview(i: number): Promise<Blob | undefined>;
  final(): Promise<Blob>;
  dispose(): void;
}

/** A diffusion model's generation: the prompt's encoding and the latents as the steps go. */
class DiffusionRun implements ImageRun {
  private readonly ort = getOrt();
  private readonly latentDims: number[];
  private x: Float32Array;
  /** Text encoder outputs the denoiser takes each step. */
  private conditioning: Record<string, Tensor> = {};
  private readonly guidance: number;
  /** The clean latents predicted by the last step. */
  private clean: Float32Array | null = null;

  private readonly pipeline: Exclude<ImagePipeline, { kind: 'janus' }>;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    readonly steps: number,
    private readonly seed: number,
    private readonly width: number,
    private readonly height: number,
    private readonly cancelled: () => boolean,
  ) {
    this.pipeline = loaded.pipeline as Exclude<ImagePipeline, { kind: 'janus' }>;
    const channels = this.pipeline.kind === 'z-image' ? 16 : 4;
    this.latentDims = [1, channels, height / 8, width / 8];
    this.x = gaussian(channels * (height / 8) * (width / 8), seed);
    this.guidance = Number(request.settings.guidance ?? 1);
    const { pipeline } = this;
    // Euler starts from noise at the largest sigma; LCM and flow matching from unit noise.
    if (pipeline.kind !== 'z-image' && pipeline.scheduler.kind === 'euler') {
      const sigma0 = eulerSchedule(steps, pipeline.scheduler.spacing).sigmas[0]!;
      this.x = this.x.map((v) => v * sigma0);
    }
  }

  private session(role: string): InferenceSession {
    return this.loaded.sessions.get(role)!;
  }

  private float(type: FloatType, values: ArrayLike<number>, dims: number[]): Tensor {
    return type === 'float16'
      ? new this.ort.Tensor('float16', toHalf(values), dims)
      : new this.ort.Tensor('float32', Float32Array.from(values), dims);
  }

  async encodePrompt(): Promise<void> {
    const { pipeline } = this;
    const prompt = this.request.prompt;
    if (pipeline.kind === 'sd') {
      const cfg = pipeline.guidance === 'cfg';
      const tokens = this.loaded.clip!.encode(prompt);
      // Classifier-free guidance: the prompt (batch 0) and the negative prompt (batch 1) together.
      const ids = cfg ? [...tokens, ...this.loaded.clip!.encode(String(this.request.settings.negative ?? ''))] : tokens;
      const out = await this.session('textEncoder').run({
        input_ids: new this.ort.Tensor('int32', Int32Array.from(ids), [cfg ? 2 : 1, 77]),
      });
      this.conditioning.encoder_hidden_states = out.last_hidden_state as Tensor;
      disposeExcept(out, ['last_hidden_state']);
      return;
    }
    if (pipeline.kind === 'sdxl') {
      const one = await this.session('textEncoder').run({
        input_ids: new this.ort.Tensor('int32', Int32Array.from(this.loaded.clip!.encode(prompt)), [1, 77]),
      });
      const two = await this.session('textEncoder2').run({
        input_ids: new this.ort.Tensor('int64', BigInt64Array.from(this.loaded.clip2!.encode(prompt), BigInt), [1, 77]),
      });
      // The two encoders' penultimate layers side by side: [1, 77, 768 + 1280].
      const a = fromHalf((one['hidden_states.11'] as Tensor).data as Uint16Array);
      const b = fromHalf((two['hidden_states.31'] as Tensor).data as Uint16Array);
      const joined = new Float32Array(77 * 2048);
      for (let t = 0; t < 77; t++) {
        joined.set(a.subarray(t * 768, (t + 1) * 768), t * 2048);
        joined.set(b.subarray(t * 1280, (t + 1) * 1280), t * 2048 + 768);
      }
      this.conditioning = {
        encoder_hidden_states: this.float('float16', joined, [1, 77, 2048]),
        text_embeds: two.text_embeds as Tensor,
        time_ids: this.float('float16', [this.height, this.width, 0, 0, this.height, this.width], [1, 6]),
      };
      disposeExcept(one, []);
      disposeExcept(two, ['text_embeds']);
      return;
    }
    const ids = zImagePromptIds(this.loaded.qwen!, prompt);
    const n = ids.length;
    const out = await this.session('textEncoder').run({
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new this.ort.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
    });
    this.conditioning.encoder_hidden_states = out.encoder_hidden_states as Tensor;
  }

  /** Denoising step `i`: predicts the clean latents and moves the latents on. */
  async step(i: number): Promise<void> {
    const { pipeline } = this;
    const result =
      pipeline.kind === 'z-image'
        ? await this.flowStep(i)
        : pipeline.scheduler.kind === 'lcm'
          ? await this.lcmStep(i)
          : await this.eulerStep(i);
    this.x = result.next;
    this.clean = result.clean;
  }

  preview(): Promise<Blob> {
    return this.decode(this.clean!, true);
  }

  final(): Promise<Blob> {
    return this.decode(this.clean!, false);
  }

  private async eulerStep(i: number): Promise<StepResult> {
    const pipeline = this.pipeline as Extract<ImagePipeline, { kind: 'sd' | 'sdxl' }>;
    const scheduler = pipeline.scheduler as Extract<typeof pipeline.scheduler, { kind: 'euler' }>;
    const { timesteps, sigmas } = eulerSchedule(this.steps, scheduler.spacing);
    const sigma = sigmas[i]!;
    const scale = 1 / Math.sqrt(sigma * sigma + 1);
    const input = this.x.map((v) => v * scale);
    const io: FloatType = pipeline.kind === 'sdxl' ? 'float16' : pipeline.io;
    const cfg = pipeline.kind === 'sd' && pipeline.guidance === 'cfg';
    const batch = cfg ? 2 : 1;
    const sample = cfg ? concat(input, input) : input;
    const t = timesteps[i]!;
    const timestep =
      pipeline.kind === 'sdxl'
        ? this.float('float16', [t], [1])
        : pipeline.timestep === 'int64'
          ? new this.ort.Tensor('int64', new BigInt64Array(batch).fill(BigInt(Math.round(t))), [batch])
          : new this.ort.Tensor('float32', new Float32Array(batch).fill(t), [batch]);
    const out = await this.session('unet').run({
      sample: this.float(io, sample, [batch, ...this.latentDims.slice(1)]),
      timestep,
      ...this.conditioning,
    });
    const raw = readFloats(out.out_sample as Tensor);
    disposeExcept(out, []);
    let eps = raw;
    if (cfg) {
      // Batch 0 is the prompt, batch 1 the negative prompt: move from the negative towards the prompt.
      const n = this.x.length;
      eps = new Float32Array(n);
      for (let k = 0; k < n; k++) eps[k] = raw[n + k]! + this.guidance * (raw[k]! - raw[n + k]!);
    }
    const clean = this.x.map((v, k) => v - sigma * eps[k]!);
    const next = sigmas[i + 1]!;
    if (scheduler.ancestral) {
      const noise = gaussian(this.x.length, this.seed + 7919 * (i + 1));
      return { clean, next: clean.map((v, k) => v + next * noise[k]!) };
    }
    return { clean, next: this.x.map((v, k) => v + (next - sigma) * eps[k]!) };
  }

  private async lcmStep(i: number): Promise<StepResult> {
    const timesteps = lcmTimesteps(this.steps);
    const acp = alphasCumprod();
    const t = timesteps[i]!;
    const out = await this.session('unet').run({
      sample: this.float('float32', this.x, this.latentDims),
      timestep: new this.ort.Tensor('float32', Float32Array.of(t), [1]),
      timestep_cond: new this.ort.Tensor('float32', guidanceEmbedding(this.guidance), [1, 256]),
      ...this.conditioning,
    });
    const eps = readFloats(out.out_sample as Tensor);
    disposeExcept(out, []);
    const alpha = acp[t]!;
    const { cSkip, cOut } = lcmScalings(t);
    const clean = this.x.map((v, k) => cOut * ((v - Math.sqrt(1 - alpha) * eps[k]!) / Math.sqrt(alpha)) + cSkip * v);
    if (i === this.steps - 1) return { clean, next: clean };
    const prev = acp[timesteps[i + 1]!]!;
    const noise = gaussian(this.x.length, this.seed + 7919 * (i + 1));
    return { clean, next: clean.map((v, k) => Math.sqrt(prev) * v + Math.sqrt(1 - prev) * noise[k]!) };
  }

  private async flowStep(i: number): Promise<StepResult> {
    const { sigmas, timesteps } = flowMatchSchedule(this.steps);
    const out = await this.session('transformer').run({
      hidden_states: this.float('float16', this.x, this.latentDims),
      timestep: this.float('float16', [timesteps[i]!], [1]),
      ...this.conditioning,
    });
    const v = readFloats(out.sample as Tensor);
    disposeExcept(out, []);
    const sigma = sigmas[i]!;
    const dt = sigmas[i + 1]! - sigma;
    return { clean: this.x.map((x, k) => x + sigma * v[k]!), next: this.x.map((x, k) => x - dt * v[k]!) };
  }

  /**
   * Latents → decoder → PNG. A preview uses the tiny decoder if there is one (it takes unscaled latents), else the
   * full decoder on latents averaged down to half size (a quarter of the work, a half-size picture).
   */
  private async decode(full: Float32Array, preview: boolean): Promise<Blob> {
    if (this.cancelled() && preview) throw new RuntimeError('cancelled', 'Generation was stopped.');
    const { pipeline } = this;
    const { sessions } = this.loaded;
    const tiny = preview && sessions.has('previewDecoder');
    const half = preview && !tiny;
    const latents = half ? halveLatents(full, this.latentDims) : full;
    const dims = half ? [1, this.latentDims[1]!, this.latentDims[2]! / 2, this.latentDims[3]! / 2] : this.latentDims;
    const width = half ? this.width / 2 : this.width;
    const height = half ? this.height / 2 : this.height;
    let input: Float32Array;
    let type: FloatType;
    if (tiny) {
      input = latents;
      type = 'float32';
    } else if (pipeline.kind === 'z-image') {
      input = latents.map((v) => v / Z_IMAGE_LATENT.scale + Z_IMAGE_LATENT.shift);
      type = 'float16';
    } else {
      input = latents.map((v) => v / (pipeline.kind === 'sdxl' ? SDXL_LATENT_SCALE : SD_LATENT_SCALE));
      type = pipeline.kind === 'sdxl' ? 'float16' : pipeline.io;
    }
    const out = await sessions.get(tiny ? 'previewDecoder' : 'vaeDecoder')!.run({ latent_sample: this.float(type, input, dims) });
    const sample = out.sample as Tensor;
    const pixels = toRgba(readFloats(sample), width, height);
    sample.dispose();
    const canvas = new OffscreenCanvas(width, height);
    canvas.getContext('2d')!.putImageData(new ImageData(pixels, width, height), 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }

  dispose(): void {
    for (const t of Object.values(this.conditioning)) t.dispose();
    this.conditioning = {};
  }
}

/** Janus's generation: 576 image tokens, a row (24) per step, sampled with guidance; previews show the rows so far. */
class JanusRun implements ImageRun {
  /** Rows of the 24 × 24 token grid: one step each. */
  readonly steps = JANUS_GRID;
  private readonly ort = getOrt();
  private readonly pipeline: Extract<ImagePipeline, { kind: 'janus' }>;
  private readonly guidance: number;
  private readonly random: () => number;
  private readonly tokens: number[] = [];
  /** The KV cache (on the GPU), per layer key then value; empty before the prompt. */
  private past: Tensor[] = [];
  private pastLength = 0;
  /** What the language model reads next: the prompt's embeddings, then each new image token's. */
  private embeds: Tensor | null = null;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    seed: number,
    private readonly cancelled: () => boolean,
  ) {
    this.pipeline = loaded.pipeline as Extract<ImagePipeline, { kind: 'janus' }>;
    this.guidance = Number(request.settings.guidance ?? 5);
    let s = seed >>> 0 || 1;
    this.random = () => {
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      return (s >>> 0) / 4294967296;
    };
  }

  private session(role: string): InferenceSession {
    return this.loaded.sessions.get(role)!;
  }

  async encodePrompt(): Promise<void> {
    const t = this.loaded.qwen!;
    const ids = janusPromptIds(t, this.request.prompt);
    // The unconditional row: the same length, everything but the first and last token blanked out.
    const pad = t.specialTokenId('<｜▁pad▁｜>');
    const blank = ids.map((id, i) => (i === 0 || i === ids.length - 1 ? id : pad));
    const out = await this.session('embedTokens').run({
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from([...ids, ...blank], BigInt), [2, ids.length]),
    });
    this.embeds = out.inputs_embeds as Tensor;
    const empty = () => new this.ort.Tensor('float16', new Uint16Array(0), [2, 16, 0, 128]);
    this.past = kvNames(this.pipeline.layers, 'past_key_values').map(empty);
  }

  /** Row `row`: 24 more tokens. */
  async step(_row: number): Promise<void> {
    for (let k = 0; k < JANUS_GRID; k++) {
      if (this.cancelled()) throw new RuntimeError('cancelled', 'Generation was stopped.');
      this.tokens.push(await this.nextToken());
    }
  }

  private async nextToken(): Promise<number> {
    const embeds = this.embeds!;
    const n = embeds.dims[1]!;
    const total = this.pastLength + n;
    const positions = BigInt64Array.from({ length: 2 * n }, (_, i) => BigInt(this.pastLength + (i % n)));
    const feeds: Record<string, Tensor> = {
      inputs_embeds: embeds,
      attention_mask: new this.ort.Tensor('int64', new BigInt64Array(2 * total).fill(1n), [2, total]),
      position_ids: new this.ort.Tensor('int64', positions, [2, n]),
    };
    kvNames(this.pipeline.layers, 'past_key_values').forEach((name, i) => (feeds[name] = this.past[i]!));
    const out = await this.session('languageModel').run(feeds);
    embeds.dispose();
    for (const t of this.past) t.dispose();
    this.past = kvNames(this.pipeline.layers, 'present').map((name) => out[name] as Tensor);
    this.pastLength = total;

    // The last position of each row → image-token logits.
    // (float16 data may come as a Float16Array: copy values, not bits.)
    const hidden = readFloats(out.hidden_states as Tensor);
    const width = 2048;
    const last = new Float32Array(2 * width);
    last.set(hidden.subarray((n - 1) * width, n * width), 0);
    last.set(hidden.subarray((2 * n - 1) * width, 2 * n * width), width);
    (out.hidden_states as Tensor).dispose();
    const head = await this.session('genHead').run({ hidden_states: new this.ort.Tensor('float16', toHalf(last), [2, 1, width]) });
    const logits = (head.logits as Tensor).data as Float32Array;
    const vocab = logits.length / 2;
    const token = this.sample(logits, vocab);
    (head.logits as Tensor).dispose();

    const next = await this.session('genImgEmbeds').run({
      image_ids: new this.ort.Tensor('int64', BigInt64Array.of(BigInt(token), BigInt(token)), [2, 1]),
    });
    this.embeds = next.inputs_embeds as Tensor;
    return token;
  }

  /** Guidance (from the blank row towards the prompt's), then a sample at temperature 1. */
  private sample(logits: Float32Array, vocab: number): number {
    const mixed = new Float32Array(vocab);
    let max = -Infinity;
    for (let i = 0; i < vocab; i++) {
      const u = logits[vocab + i]!;
      mixed[i] = u + this.guidance * (logits[i]! - u);
      if (mixed[i]! > max) max = mixed[i]!;
    }
    let sum = 0;
    for (let i = 0; i < vocab; i++) sum += mixed[i] = Math.exp(mixed[i]! - max);
    let r = this.random() * sum;
    for (let i = 0; i < vocab; i++) {
      r -= mixed[i]!;
      if (r <= 0) return i;
    }
    return vocab - 1;
  }

  /** The rows so far, every few rows (decoding costs about as much as two rows); the rest is left grey. */
  async preview(row: number): Promise<Blob | undefined> {
    return (row + 1) % JANUS_PREVIEW_ROWS === 0 ? this.decode((row + 1) * JANUS_GRID) : undefined;
  }

  final(): Promise<Blob> {
    return this.decode(JANUS_GRID * JANUS_GRID);
  }

  private async decode(known: number): Promise<Blob> {
    const total = JANUS_GRID * JANUS_GRID;
    const tokens = [...this.tokens.slice(0, known)];
    while (tokens.length < total) tokens.push(tokens[tokens.length - JANUS_GRID] ?? tokens[tokens.length - 1] ?? 0);
    const out = await this.session('imageDecode').run({
      generated_tokens: new this.ort.Tensor('int64', BigInt64Array.from(tokens, BigInt), [1, total]),
    });
    const image = out.decoded_image as Tensor;
    const size = image.dims[3]!;
    const pixels = toRgba(readFloats(image), size, size);
    image.dispose();
    const done = Math.floor(known / JANUS_GRID) * (size / JANUS_GRID);
    pixels.fill(128, done * size * 4);
    for (let i = done * size * 4 + 3; i < pixels.length; i += 4) pixels[i] = 255;
    const canvas = new OffscreenCanvas(size, size);
    canvas.getContext('2d')!.putImageData(new ImageData(pixels, size, size), 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }

  dispose(): void {
    for (const t of this.past) t.dispose();
    this.past = [];
    this.embeds?.dispose();
    this.embeds = null;
  }
}

/** KV tensor names of a language model's layers: `<prefix>.<i>.key` and `.value`. */
function kvNames(layers: number, prefix: 'past_key_values' | 'present'): string[] {
  return Array.from({ length: layers }, (_, i) => [`${prefix}.${i}.key`, `${prefix}.${i}.value`]).flat();
}

function tokenizers(pipeline: ImagePipeline, files: ImageFileSource): Pick<Loaded, 'clip' | 'clip2' | 'qwen'> {
  const json = (file: ArtifactFile): unknown => JSON.parse(new TextDecoder().decode(files.bytes(file)));
  const clip = (t: ClipTokenizerFiles) =>
    t.json
      ? ClipTokenizer.fromJSON(json(t.json), t.pad)
      : new ClipTokenizer(json(t.vocab!) as Record<string, number>, new TextDecoder().decode(files.bytes(t.merges!)), t.pad);
  if (pipeline.kind === 'z-image' || pipeline.kind === 'janus') return { qwen: BpeTokenizer.fromJSON(json(pipeline.tokenizer)) };
  if (pipeline.kind === 'sdxl') return { clip: clip(pipeline.tokenizer), clip2: clip(pipeline.tokenizer2) };
  return { clip: clip(pipeline.tokenizer) };
}

async function release(loaded: Loaded): Promise<void> {
  await Promise.all([...loaded.sessions.values()].map((s) => s.release().catch(() => {})));
  loaded.sessions.clear();
}

function readFloats(tensor: Tensor): Float32Array {
  return tensor.type === 'float16' ? fromHalf(tensor.data as Uint16Array) : (tensor.data as Float32Array);
}

function disposeExcept(outputs: InferenceSession.OnnxValueMapType, keep: string[]): void {
  for (const [name, value] of Object.entries(outputs)) if (!keep.includes(name)) (value as Tensor).dispose();
}

/** [1, C, h, w] latents averaged over 2 × 2 blocks: [1, C, h/2, w/2]. */
function halveLatents(latents: Float32Array, dims: number[]): Float32Array {
  const [, c, h, w] = dims as [number, number, number, number];
  const out = new Float32Array(c * (h / 2) * (w / 2));
  for (let ch = 0; ch < c; ch++) {
    for (let y = 0; y < h / 2; y++) {
      for (let x = 0; x < w / 2; x++) {
        const i = ch * h * w + 2 * y * w + 2 * x;
        out[ch * (h / 2) * (w / 2) + y * (w / 2) + x] = (latents[i]! + latents[i + 1]! + latents[i + w]! + latents[i + w + 1]!) / 4;
      }
    }
  }
  return out;
}

function concat(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

function bytesOf(g: { graph: ArtifactFile; data?: ArtifactFile[] }): number {
  return g.graph.sizeBytes + (g.data ?? []).reduce((n, f) => n + f.sizeBytes, 0);
}

/** The name a graph refers to its weight file by. */
function fileName(file: ArtifactFile): string {
  const path = file.from?.path ?? file.path;
  return path.slice(path.lastIndexOf('/') + 1);
}

const OUT_OF_MEMORY = /out of memory|\bOOM\b|memory copy|allocat|GPUOutOfMemoryError/i;

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
