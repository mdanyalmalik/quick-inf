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
// - 'sd3' (Stable Diffusion 3.5): CLIP-L, CLIP-G and T5 text features → an MMDiT transformer over flow matching
//   (with guidance, as SD 1.5) → VAE decoder (16 latent channels).
// - 'flux1' (FLUX.1 schnell, dev): CLIP-L's pooled output and T5's tokens → a rectified-flow transformer over latents
//   packed 2 × 2 into tokens → VAE decoder (16 latent channels).
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
import { UnigramTokenizer } from '../../tokenizer/unigram-tokenizer';
import { RuntimeError } from '../contract';
import type { ByteSource } from '../onnx/external-weights';
import { createGraphSession } from '../onnx/onnx-adapter';
import { getOrt } from '../onnx/ort-env';
import { flux2PromptIds, flux3PromptIds, janusPromptIds, zImagePromptIds } from './prompts';
import {
  alphasCumprod,
  amusedStep,
  eulerSchedule,
  flowMatchSchedule,
  flux1Schedule,
  flux2Schedule,
  flux3Schedule,
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
  t5?: UnigramTokenizer;
}

const SD_LATENT_SCALE = 0.18215;
const SDXL_LATENT_SCALE = 0.13025;
const Z_IMAGE_LATENT = { scale: 0.3611, shift: 0.1159 };
const SD3_LATENT = { scale: 1.5305, shift: 0.0609 };
const FLUX1_LATENT = { scale: 0.3611, shift: 0.1159 };
/** SD 3.5's text: 77 CLIP positions, then T5's (diffusers' default length for it). */
const SD3_CLIP_TOKENS = 77;
const SD3_T5_TOKENS = 256;
const SD3_WIDTH = 4096;
const FLUX2_CHANNELS = 128;
const FLUX2_TEXT_TOKENS = 512;
const FLUX2_ATTENTION_CHUNK = 1024;
/** Janus's image is a 24 × 24 grid of tokens. */
const JANUS_GRID = 24;
const JANUS_PREVIEW_ROWS = 4;

export class ImageEngine {
  private loaded: Loaded | null = null;
  private cancelRequested = false;

  get modelId(): string | null {
    return this.loaded?.model.id ?? null;
  }

  /** `processor` 'cpu': ONNX Runtime's WebAssembly engine instead of WebGPU (devices without a usable GPU). */
  async load(model: ImageModelSpec, files: ImageFileSource, processor: 'gpu' | 'cpu' = 'gpu'): Promise<{ sessionMs: number }> {
    await this.unload();
    const pipeline = model.pipeline!;
    const start = performance.now();
    const loaded: Loaded = { model, pipeline, sessions: new Map(), ...tokenizers(pipeline, files) };
    let device: GPUDevice | null = null;
    if (processor === 'gpu') {
      try {
        device = await getGpuDevice();
      } catch (err) {
        throw new RuntimeError('load-failed', `The GPU could not be opened: ${message(err)}`, true);
      }
    }
    try {
      // Smallest first, so a failure shows before the large graphs are read.
      const graphs = Object.entries(imageGraphs(pipeline)).sort(([, a], [, b]) => bytesOf(a) - bytesOf(b));
      for (const [role, g] of graphs) {
        const data = Object.fromEntries((g.data ?? []).map((f) => [fileName(f), files.source(f)]));
        // Janus's KV cache stays on the GPU between tokens.
        const gpuOutputs = pipeline.kind === 'janus' && role === 'languageModel' ? kvNames(pipeline.layers, 'present') : [];
        loaded.sessions.set(
          role,
          await createGraphSession(files.source(g.graph), device, data, gpuOutputs, {
            matMulNBitsAccuracy: g.matMulNBitsAccuracy,
            optimization: g.optimization,
            freeDimensions: g.freeDimensions,
            quantizeWeights: g.quantizeWeights,
            computeFloat32: g.computeFloat32,
          }),
        );
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
      throw new RuntimeError('load-failed', `The model could not be loaded on the ${processor === 'cpu' ? 'CPU' : 'GPU'}: ${message(err)}`, true);
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
        : loaded.pipeline.kind === 'amused'
        ? new AmusedRun(loaded, request, Math.max(2, Math.min(24, Math.round(Number(settings.steps ?? 12)))), seed, width, height)
        : loaded.pipeline.kind === 'flux1'
          ? new Flux1Run(
              loaded,
              request,
              Math.max(1, Math.min(50, Math.round(Number(settings.steps ?? loaded.model.steps ?? 4)))),
              seed,
              width,
              height,
              cancelled,
            )
          : loaded.pipeline.kind === 'sd3'
          ? new Sd3Run(
              loaded,
              request,
              Math.max(1, Math.min(50, Math.round(Number(settings.steps ?? loaded.model.steps ?? 28)))),
              seed,
              width,
              height,
              cancelled,
            )
          : loaded.pipeline.kind === 'flux3'
          ? new Flux3Run(
              loaded,
              request,
              Math.max(1, Math.min(32, Math.round(Number(settings.steps ?? loaded.model.steps ?? 16)))),
              seed,
              width,
              height,
              cancelled,
            )
          : loaded.pipeline.kind === 'flux2'
          ? new Flux2Run(
              loaded,
              request,
              Math.max(1, Math.min(12, Math.round(Number(settings.steps ?? loaded.model.steps ?? 4)))),
              seed,
              width,
              height,
              cancelled,
            )
          : new DiffusionRun(
              loaded,
              request,
              Math.max(1, Math.min(50, Math.round(Number(settings.steps ?? loaded.model.steps ?? 1)))),
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

  private readonly pipeline: Exclude<ImagePipeline, { kind: 'janus' | 'flux2' | 'flux3' | 'sd3' | 'flux1' | 'amused' }>;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    readonly steps: number,
    private readonly seed: number,
    private readonly width: number,
    private readonly height: number,
    private readonly cancelled: () => boolean,
  ) {
    this.pipeline = loaded.pipeline as Exclude<ImagePipeline, { kind: 'janus' | 'flux2' | 'flux3' | 'sd3' | 'flux1' | 'amused' }>;
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
      const out = await this.session('textEncoder').run({ input_ids: this.ids(ids, [cfg ? 2 : 1, 77], pipeline.textInput ?? 'int32') });
      this.conditioning.encoder_hidden_states = out.last_hidden_state as Tensor;
      disposeExcept(out, ['last_hidden_state']);
      return;
    }
    if (pipeline.kind === 'sdxl') {
      const io = pipeline.io ?? 'float16';
      const positive = await this.sdxlEncode(prompt);
      if (pipeline.guidance !== 'cfg') {
        this.conditioning = {
          encoder_hidden_states: this.float(io, positive.hidden, [1, 77, 2048]),
          text_embeds: this.float(io, positive.pooled, [1, 1280]),
          time_ids: this.float(io, this.timeIds(), [1, 6]),
        };
        return;
      }
      // Guidance: the prompt (batch 0) and the negative prompt (batch 1). An empty negative prompt is all zeros,
      // as SDXL's pipeline does (force_zeros_for_empty_prompt).
      const negativeText = String(this.request.settings.negative ?? '').trim();
      const negative = negativeText
        ? await this.sdxlEncode(negativeText)
        : { hidden: new Float32Array(77 * 2048), pooled: new Float32Array(1280) };
      this.conditioning = {
        encoder_hidden_states: this.float(io, concat(positive.hidden, negative.hidden), [2, 77, 2048]),
        text_embeds: this.float(io, concat(positive.pooled, negative.pooled), [2, 1280]),
        time_ids: this.float(io, [...this.timeIds(), ...this.timeIds()], [2, 6]),
      };
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

  /** SDXL's text: both encoders' penultimate layers side by side ([77, 768 + 1280]) and the second's pooled output. */
  private async sdxlEncode(text: string): Promise<{ hidden: Float32Array; pooled: Float32Array }> {
    const one = await this.session('textEncoder').run({ input_ids: this.ids(this.loaded.clip!.encode(text), [1, 77], 'int32') });
    const two = await this.session('textEncoder2').run({ input_ids: this.ids(this.loaded.clip2!.encode(text), [1, 77], 'int64') });
    const a = readFloats(one['hidden_states.11'] as Tensor);
    const b = readFloats(two['hidden_states.31'] as Tensor);
    const pooled = Float32Array.from(readFloats(two.text_embeds as Tensor));
    disposeExcept(one, []);
    disposeExcept(two, []);
    const hidden = new Float32Array(77 * 2048);
    for (let t = 0; t < 77; t++) {
      hidden.set(a.subarray(t * 768, (t + 1) * 768), t * 2048);
      hidden.set(b.subarray(t * 1280, (t + 1) * 1280), t * 2048 + 768);
    }
    return { hidden, pooled };
  }

  /** SDXL's size conditioning: original size, crop (none), target size. */
  private timeIds(): number[] {
    return [this.height, this.width, 0, 0, this.height, this.width];
  }

  private ids(ids: number[], dims: number[], type: 'int32' | 'int64'): Tensor {
    return type === 'int64'
      ? new this.ort.Tensor('int64', BigInt64Array.from(ids, BigInt), dims)
      : new this.ort.Tensor('int32', Int32Array.from(ids), dims);
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
    const io: FloatType = pipeline.kind === 'sdxl' ? (pipeline.io ?? 'float16') : pipeline.io;
    const cfg = pipeline.guidance === 'cfg';
    const batch = cfg ? 2 : 1;
    const sample = cfg ? concat(input, input) : input;
    const t = timesteps[i]!;
    const timestepType = pipeline.kind === 'sdxl' ? (pipeline.timestep ?? 'float16') : pipeline.timestep;
    // One timestep per batch row, or one for all as a scalar.
    const scalar = pipeline.kind === 'sd' && pipeline.scalarTimestep;
    const n = scalar ? 1 : batch;
    const tDims = scalar ? [] : [batch];
    const timestep =
      timestepType === 'float16'
        ? this.float('float16', new Array<number>(n).fill(t), tDims)
        : timestepType === 'int64'
          ? new this.ort.Tensor('int64', new BigInt64Array(n).fill(BigInt(Math.round(t))), tDims)
          : new this.ort.Tensor('float32', new Float32Array(n).fill(t), tDims);
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
      ...((this.pipeline as Extract<ImagePipeline, { kind: 'sd' }>).guidance === 'embedding' && {
        timestep_cond: new this.ort.Tensor('float32', guidanceEmbedding(this.guidance), [1, 256]),
      }),
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
      const scale = pipeline.kind === 'sd' && pipeline.vaeScales ? 1 : pipeline.kind === 'sdxl' ? SDXL_LATENT_SCALE : SD_LATENT_SCALE;
      input = scale === 1 ? latents : latents.map((v) => v / scale);
      type = pipeline.kind === 'sdxl' ? (pipeline.io ?? 'float16') : pipeline.io;
    }
    const names = !tiny && pipeline.kind === 'sd' && pipeline.vaeNames ? pipeline.vaeNames : { input: 'latent_sample', output: 'sample' };
    const out = await sessions.get(tiny ? 'previewDecoder' : 'vaeDecoder')!.run({ [names.input]: this.float(type, input, dims) });
    return toPng(out[names.output] as Tensor, width, height);
  }

  dispose(): void {
    for (const t of Object.values(this.conditioning)) t.dispose();
    this.conditioning = {};
  }
}

/**
 * Stable Diffusion 3.5's generation: the three text encoders' features, then flow matching over [1, 16, h/8, w/8]
 * latents (the transformer predicts the velocity towards the noise; guidance mixes the prompt's and the negative
 * prompt's), then the decoder. Previews decode latents averaged down to half size, as for SD 1.5. The prompt and the
 * negative prompt run one after the other: the AMD exports fold each block's modulation into LayerNormalization's
 * scale and bias, which only fits a batch of one.
 */
class Sd3Run implements ImageRun {
  private readonly ort = getOrt();
  private readonly pipeline: Extract<ImagePipeline, { kind: 'sd3' }>;
  private readonly latentDims: number[];
  private x: Float32Array;
  private readonly sigmas: number[];
  private readonly guidance: number;
  private readonly cfg: boolean;
  /** The prompt's text inputs, then the negative prompt's (with guidance). */
  private conditioning: Record<string, Tensor>[] = [];
  private clean: Float32Array | null = null;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    readonly steps: number,
    seed: number,
    private readonly width: number,
    private readonly height: number,
    private readonly cancelled: () => boolean,
  ) {
    this.pipeline = loaded.pipeline as Extract<ImagePipeline, { kind: 'sd3' }>;
    this.latentDims = [1, 16, height / 8, width / 8];
    this.x = gaussian(16 * (height / 8) * (width / 8), seed);
    this.sigmas = flowMatchSchedule(steps, 3).sigmas;
    this.guidance = Number(request.settings.guidance ?? 1);
    this.cfg = this.pipeline.guidance === 'cfg' && this.guidance !== 1;
  }

  private session(role: string): InferenceSession {
    return this.loaded.sessions.get(role)!;
  }

  async encodePrompt(): Promise<void> {
    const io = inputType(this.session('transformer'), 'hidden_states');
    const tokens = SD3_CLIP_TOKENS + SD3_T5_TOKENS;
    // With guidance, the negative prompt too (encoded even when empty, as diffusers does).
    const texts = this.cfg ? [this.request.prompt, String(this.request.settings.negative ?? '')] : [this.request.prompt];
    for (const text of texts) {
      const { hidden, pooled } = await this.encode(text);
      this.conditioning.push({
        encoder_hidden_states: floatTensor(this.ort, io, hidden, [1, tokens, SD3_WIDTH]),
        pooled_projections: floatTensor(this.ort, io, pooled, [1, pooled.length]),
      });
    }
  }

  /** [333, 4096] text features (CLIP-L | CLIP-G | zeros for 77 rows, then T5's 256) and [768 + 1280] pooled. */
  private async encode(text: string): Promise<{ hidden: Float32Array; pooled: Float32Array }> {
    const clip = async (role: string, tokenizer: ClipTokenizer) => {
      const session = this.session(role);
      const out = await session.run({ input_ids: idsTensor(this.ort, session, tokenizer.encode(text), [1, SD3_CLIP_TOKENS]) });
      // The penultimate hidden state, and the pooled (projected) output.
      const layers = session.outputNames.filter((n) => /hidden_states\d*\.\d+$/.test(n));
      const last = Math.max(...layers.map((n) => Number(n.slice(n.lastIndexOf('.') + 1))));
      const hiddenName = layers.find((n) => n.endsWith(`.${last - 1}`))!;
      // The pooled output is the one [1, width] output besides the layers (by shape: the AMD export's CLIP-L calls it
      // last_hidden_state_1, and its [1, 77, 768] last layer text_embeds_1).
      const pooledName = session.outputNames.find((n) => !layers.includes(n) && (out[n] as Tensor).dims.length === 2)!;
      const hidden = Float32Array.from(readFloats(out[hiddenName] as Tensor));
      const pooled = Float32Array.from(readFloats(out[pooledName] as Tensor));
      disposeExcept(out, []);
      return { hidden, pooled };
    };
    const l = await clip('textEncoder', this.loaded.clip!);
    const g = await clip('textEncoder2', this.loaded.clip2!);
    const t5Session = this.session('textEncoder3');
    const t5Out = await t5Session.run({
      input_ids: idsTensor(this.ort, t5Session, this.loaded.t5!.encodePadded(text, SD3_T5_TOKENS), [1, SD3_T5_TOKENS]),
    });
    const t5 = readFloats(t5Out[t5Session.outputNames[0]!] as Tensor);
    const widthL = l.hidden.length / SD3_CLIP_TOKENS;
    const widthG = g.hidden.length / SD3_CLIP_TOKENS;
    const hidden = new Float32Array((SD3_CLIP_TOKENS + SD3_T5_TOKENS) * SD3_WIDTH);
    for (let t = 0; t < SD3_CLIP_TOKENS; t++) {
      hidden.set(l.hidden.subarray(t * widthL, (t + 1) * widthL), t * SD3_WIDTH);
      hidden.set(g.hidden.subarray(t * widthG, (t + 1) * widthG), t * SD3_WIDTH + widthL);
    }
    hidden.set(t5.subarray(0, SD3_T5_TOKENS * SD3_WIDTH), SD3_CLIP_TOKENS * SD3_WIDTH);
    disposeExcept(t5Out, []);
    return { hidden, pooled: concat(l.pooled, g.pooled) };
  }

  async step(i: number): Promise<void> {
    const session = this.session('transformer');
    const io = inputType(session, 'hidden_states');
    const sigma = this.sigmas[i]!;
    const velocities: Float32Array[] = [];
    for (const conditioning of this.conditioning) {
      if (this.cancelled()) throw new RuntimeError('cancelled', 'Generation was stopped.');
      const out = await session.run({
        hidden_states: floatTensor(this.ort, io, this.x, this.latentDims),
        timestep: floatTensor(this.ort, inputType(session, 'timestep'), [sigma * 1000], [1]),
        ...conditioning,
      });
      velocities.push(Float32Array.from(readFloats(out[session.outputNames[0]!] as Tensor)));
      disposeExcept(out, []);
    }
    const n = this.x.length;
    let v = velocities[0]!;
    if (this.cfg) {
      // From the negative prompt's prediction towards the prompt's.
      const [cond, uncond] = velocities as [Float32Array, Float32Array];
      v = new Float32Array(n);
      for (let k = 0; k < n; k++) v[k] = uncond[k]! + this.guidance * (cond[k]! - uncond[k]!);
    }
    const dt = this.sigmas[i + 1]! - sigma;
    this.clean = this.x.map((x, k) => x - sigma * v[k]!);
    this.x = this.x.map((x, k) => x + dt * v[k]!);
  }

  preview(): Promise<Blob> {
    return this.decode(this.clean!, true);
  }

  final(): Promise<Blob> {
    return this.decode(this.clean!, false);
  }

  private async decode(latents: Float32Array, preview: boolean): Promise<Blob> {
    if (this.cancelled() && preview) throw new RuntimeError('cancelled', 'Generation was stopped.');
    const input = (preview ? halveLatents(latents, this.latentDims) : latents).map((v) => v / SD3_LATENT.scale + SD3_LATENT.shift);
    const [, c, h, w] = this.latentDims as [number, number, number, number];
    const dims = preview ? [1, c, h / 2, w / 2] : this.latentDims;
    const session = this.session('vaeDecoder');
    const out = await session.run({ latent_sample: floatTensor(this.ort, inputType(session, 'latent_sample'), input, dims) });
    const scale = preview ? 2 : 1;
    return toPng(out[session.outputNames[0]!] as Tensor, this.width / scale, this.height / scale);
  }

  dispose(): void {
    for (const c of this.conditioning) for (const t of Object.values(c)) t.dispose();
    this.conditioning = [];
  }
}

/**
 * FLUX.1's generation: [tokens, 64] latents (each token 2 × 2 latent pixels × 16 channels) moved along the
 * transformer's velocity from noise (σ = 1) to the image (σ = 0); dev's guidance goes in as an input.
 */
class Flux1Run implements ImageRun {
  private readonly ort = getOrt();
  private readonly pipeline: Extract<ImagePipeline, { kind: 'flux1' }>;
  private readonly rows: number;
  private readonly cols: number;
  private x: Float32Array;
  private readonly sigmas: number[];
  private conditioning: Record<string, Tensor> = {};
  private clean: Float32Array | null = null;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    readonly steps: number,
    seed: number,
    private readonly width: number,
    private readonly height: number,
    private readonly cancelled: () => boolean,
  ) {
    this.pipeline = loaded.pipeline as Extract<ImagePipeline, { kind: 'flux1' }>;
    this.rows = height / 16;
    this.cols = width / 16;
    this.x = gaussian(this.rows * this.cols * 64, seed);
    this.sigmas = flux1Schedule(steps, this.rows * this.cols, !!this.pipeline.shiftSchedule);
  }

  private session(role: string): InferenceSession {
    return this.loaded.sessions.get(role)!;
  }

  async encodePrompt(): Promise<void> {
    const text = this.request.prompt;
    const clip = this.session('textEncoder');
    const clipOut = await clip.run({ input_ids: idsTensor(this.ort, clip, this.loaded.clip!.encode(text), [1, 77]) });
    const pooled = Float32Array.from(readFloats(clipOut.pooler_output as Tensor));
    disposeExcept(clipOut, []);
    const t5 = this.session('textEncoder2');
    const tokens = this.pipeline.textTokens;
    const t5Out = await t5.run({ input_ids: idsTensor(this.ort, t5, this.loaded.t5!.encodePadded(text, tokens), [1, tokens]) });
    const hidden = Float32Array.from(readFloats(t5Out[t5.outputNames[0]!] as Tensor));
    disposeExcept(t5Out, []);
    const transformer = this.session('transformer');
    const io = inputType(transformer, 'hidden_states');
    const imageIds = new BigInt64Array(this.rows * this.cols * 3);
    for (let r = 0, k = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++, k += 3) imageIds.set([0n, BigInt(r), BigInt(c)], k);
    this.conditioning = {
      encoder_hidden_states: floatTensor(this.ort, io, hidden, [1, tokens, hidden.length / tokens]),
      pooled_projections: floatTensor(this.ort, io, pooled, [1, pooled.length]),
      img_ids: new this.ort.Tensor('int64', imageIds, [this.rows * this.cols, 3]),
      txt_ids: new this.ort.Tensor('int64', new BigInt64Array(tokens * 3), [tokens, 3]),
      ...(this.pipeline.guidanceInput && {
        guidance: floatTensor(this.ort, inputType(transformer, 'guidance'), [Number(this.request.settings.guidance ?? 3.5)], [1]),
      }),
    };
  }

  async step(i: number): Promise<void> {
    const session = this.session('transformer');
    const sigma = this.sigmas[i]!;
    const out = await session.run({
      hidden_states: floatTensor(this.ort, inputType(session, 'hidden_states'), this.x, [1, this.rows * this.cols, 64]),
      timestep: floatTensor(this.ort, inputType(session, 'timestep'), [sigma], [1]),
      ...this.conditioning,
    });
    const v = readFloats(out[session.outputNames[0]!] as Tensor);
    disposeExcept(out, []);
    const dt = this.sigmas[i + 1]! - sigma;
    this.clean = this.x.map((x, k) => x - sigma * v[k]!);
    this.x = this.x.map((x, k) => x + dt * v[k]!);
  }

  preview(): Promise<Blob> {
    return this.decode(this.clean!, true);
  }

  final(): Promise<Blob> {
    return this.decode(this.clean!, false);
  }

  /** [tokens, 64] → [1, 16, h/8, w/8] (diffusers' _unpack_latents), scaled for the decoder; previews at half size. */
  private async decode(packed: Float32Array, preview: boolean): Promise<Blob> {
    if (this.cancelled() && preview) throw new RuntimeError('cancelled', 'Generation was stopped.');
    const h = this.rows * 2;
    const w = this.cols * 2;
    const latents = new Float32Array(16 * h * w);
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < this.cols; c++) {
        const token = (r * this.cols + c) * 64;
        for (let ch = 0; ch < 16; ch++)
          for (let dy = 0; dy < 2; dy++)
            for (let dx = 0; dx < 2; dx++) latents[ch * h * w + (2 * r + dy) * w + 2 * c + dx] = packed[token + ch * 4 + dy * 2 + dx]!;
      }
    const dims = [1, 16, h, w];
    const input = (preview ? halveLatents(latents, dims) : latents).map((v) => v / FLUX1_LATENT.scale + FLUX1_LATENT.shift);
    const session = this.session('vaeDecoder');
    const out = await session.run({
      latent_sample: floatTensor(this.ort, inputType(session, 'latent_sample'), input, preview ? [1, 16, h / 2, w / 2] : dims),
    });
    const scale = preview ? 2 : 1;
    return toPng(out[session.outputNames[0]!] as Tensor, this.width / scale, this.height / scale);
  }

  dispose(): void {
    for (const t of Object.values(this.conditioning)) t.dispose();
    this.conditioning = {};
  }
}

/** A graph input's float type (float16 unless it says float32). */
function inputType(session: InferenceSession, name: string): FloatType {
  const meta = session.inputMetadata.find((m) => m.name === name);
  return meta?.isTensor && meta.type === 'float32' ? 'float32' : 'float16';
}

function floatTensor(ort: ReturnType<typeof getOrt>, type: FloatType, values: ArrayLike<number>, dims: number[]): Tensor {
  return type === 'float16' ? new ort.Tensor('float16', toHalf(values), dims) : new ort.Tensor('float32', Float32Array.from(values), dims);
}

/** Token ids as the graph's input_ids type (int32 or int64). */
function idsTensor(ort: ReturnType<typeof getOrt>, session: InferenceSession, ids: number[], dims: number[]): Tensor {
  const meta = session.inputMetadata.find((m) => m.name === 'input_ids');
  return meta?.isTensor && meta.type === 'int32'
    ? new ort.Tensor('int32', Int32Array.from(ids), dims)
    : new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), dims);
}

/** A [1, 3, h, w] image in [−1, 1] as a PNG. */
function toPng(sample: Tensor, width: number, height: number): Promise<Blob> {
  const pixels = toRgba(readFloats(sample), width, height);
  sample.dispose();
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext('2d')!.putImageData(new ImageData(pixels, width, height), 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
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
    // The cache's shape from the graph: [batch, KV heads, positions, head size].
    const meta = this.session('languageModel').inputMetadata.find((m) => m.name === 'past_key_values.0.key');
    const [, heads, , size] = meta?.isTensor ? meta.shape : [];
    const empty = () => new this.ort.Tensor('float16', new Uint16Array(0), [2, Number(heads ?? 16), 0, Number(size ?? 128)]);
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
    };
    // Exports with attention that works out positions itself (GroupQueryAttention) take no position ids.
    if (this.session('languageModel').inputNames.includes('position_ids')) feeds.position_ids = new this.ort.Tensor('int64', positions, [2, n]);
    kvNames(this.pipeline.layers, 'past_key_values').forEach((name, i) => (feeds[name] = this.past[i]!));
    const out = await this.session('languageModel').run(feeds);
    embeds.dispose();
    for (const t of this.past) t.dispose();
    this.past = kvNames(this.pipeline.layers, 'present').map((name) => out[name] as Tensor);
    this.pastLength = total;

    // The last position of each row → image-token logits.
    // (float16 data may come as a Float16Array: copy values, not bits.)
    const hidden = readFloats(out.hidden_states as Tensor);
    const width = embeds.dims[2]!;
    const last = new Float32Array(2 * width);
    last.set(hidden.subarray((n - 1) * width, n * width), 0);
    last.set(hidden.subarray((2 * n - 1) * width, 2 * n * width), width);
    (out.hidden_states as Tensor).dispose();
    const head = await this.session('genHead').run({ hidden_states: new this.ort.Tensor('float16', toHalf(last), [2, 1, width]) });
    const logits = readFloats(head.logits as Tensor);
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

/**
 * aMUSEd's generation: a grid of image tokens (one per 16 × 16 pixels), all masked at first; each step predicts every
 * masked token with guidance and keeps the most confident (amusedStep). Previews decode the prediction so far.
 */
class AmusedRun implements ImageRun {
  private readonly ort = getOrt();
  private readonly pipeline: Extract<ImagePipeline, { kind: 'amused' }>;
  private readonly guidance: number;
  private readonly random: () => number;
  private readonly rows: number;
  private readonly cols: number;
  private tokens: Int32Array;
  private predicted: Int32Array;
  /** The negative prompt's and the prompt's encodings, as one batch of 2. */
  private conditioning: { hidden: Tensor; pooled: Tensor } | null = null;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    readonly steps: number,
    seed: number,
    private readonly width: number,
    private readonly height: number,
  ) {
    this.pipeline = loaded.pipeline as Extract<ImagePipeline, { kind: 'amused' }>;
    this.guidance = Number(request.settings.guidance ?? 10);
    this.rows = height / 16;
    this.cols = width / 16;
    this.tokens = new Int32Array(this.rows * this.cols).fill(this.pipeline.maskToken);
    this.predicted = this.tokens;
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
    const clip = this.loaded.clip!;
    const ids = [...clip.encode(String(this.request.settings.negative ?? '')), ...clip.encode(this.request.prompt)];
    const out = await this.session('textEncoder').run({ input_ids: this.idsTensor(ids, [2, 77]) });
    this.conditioning = { hidden: out.encoder_hidden_states as Tensor, pooled: out.text_embeds as Tensor };
  }

  async step(i: number): Promise<void> {
    const { hidden, pooled } = this.conditioning!;
    const out = await this.session('transformer').run({
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from(this.tokens, BigInt), [1, this.rows, this.cols]),
      micro_conds: new this.ort.Tensor('float32', Float32Array.of(this.width, this.height, 0, 0, 6), [1, 5]),
      text_embeds: pooled,
      encoder_hidden_states: hidden,
      guidance: new this.ort.Tensor('float32', Float32Array.of(this.guidance), [1]),
    });
    const logits = readFloats(out.logits as Tensor);
    (out.logits as Tensor).dispose();
    const { next, predicted } = amusedStep(logits, this.tokens, i, this.steps, this.pipeline.maskToken, this.random);
    this.tokens = next;
    this.predicted = predicted;
  }

  preview(): Promise<Blob | undefined> {
    return this.decode(this.predicted);
  }

  final(): Promise<Blob> {
    return this.decode(this.tokens);
  }

  private async decode(tokens: Int32Array): Promise<Blob> {
    const out = await this.session('vqDecoder').run({
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from(tokens, BigInt), [1, this.rows, this.cols]),
    });
    return toPng(out.sample as Tensor, this.width, this.height);
  }

  private idsTensor(ids: number[], dims: number[]): Tensor {
    return idsTensor(this.ort, this.session('textEncoder'), ids, dims);
  }

  dispose(): void {
    this.conditioning?.hidden.dispose();
    this.conditioning?.pooled.dispose();
    this.conditioning = null;
  }
}

/**
 * FLUX.2 klein's generation (as the flux2-vulkan-bridge runtime its export was made for): flow matching over
 * [tokens, 128] latents, one token per 16 × 16 pixels, with a schedule shifted for the image size; the decoder runs
 * in stages, its attention in chunks of 1,024 positions.
 */
class Flux2Run implements ImageRun {
  private readonly ort = getOrt();
  private readonly latentW: number;
  private readonly latentH: number;
  private readonly tokens: number;
  private x: Float32Array;
  private readonly schedule: number[];
  private context: Tensor | null = null;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    readonly steps: number,
    seed: number,
    private readonly width: number,
    private readonly height: number,
    private readonly cancelled: () => boolean,
  ) {
    this.latentW = width / 16;
    this.latentH = height / 16;
    this.tokens = this.latentW * this.latentH;
    this.x = gaussian(this.tokens * FLUX2_CHANNELS, seed);
    this.schedule = flux2Schedule(steps, this.tokens);
  }

  private session(role: string): InferenceSession {
    return this.loaded.sessions.get(role)!;
  }

  async encodePrompt(): Promise<void> {
    const { ids, mask } = flux2PromptIds(this.loaded.qwen!, this.request.prompt, FLUX2_TEXT_TOKENS);
    const out = await this.session('textEncoder').run({
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, FLUX2_TEXT_TOKENS]),
      attention_mask: new this.ort.Tensor('int64', BigInt64Array.from(mask, BigInt), [1, FLUX2_TEXT_TOKENS]),
    });
    this.context = out.ctx as Tensor;
  }

  async step(i: number): Promise<void> {
    const imageIds = new Float32Array(this.tokens * 4);
    for (let h = 0, k = 0; h < this.latentH; h++) for (let w = 0; w < this.latentW; w++, k += 4) imageIds.set([0, h, w, 0], k);
    const textIds = new Float32Array(FLUX2_TEXT_TOKENS * 4);
    for (let t = 0; t < FLUX2_TEXT_TOKENS; t++) textIds[t * 4 + 3] = t;
    const out = await this.session('transformer').run({
      x: new this.ort.Tensor('float16', toHalf(this.x), [1, this.tokens, FLUX2_CHANNELS]),
      x_ids: new this.ort.Tensor('float32', imageIds, [1, this.tokens, 4]),
      timesteps: new this.ort.Tensor('float16', toHalf([this.schedule[i]!]), [1]),
      ctx: this.context!,
      ctx_ids: new this.ort.Tensor('float32', textIds, [1, FLUX2_TEXT_TOKENS, 4]),
    });
    const pred = readFloats(out.pred as Tensor);
    disposeExcept(out, []);
    const dt = this.schedule[i + 1]! - this.schedule[i]!;
    this.x = this.x.map((v, k) => v + dt * pred[k]!);
  }

  /** The decoder is too slow to run after every step; the reveal comes at the end. */
  async preview(): Promise<undefined> {
    return undefined;
  }

  async final(): Promise<Blob> {
    // [tokens, 128] → [1, 128, h/16, w/16].
    const z = new Float32Array(this.x.length);
    for (let t = 0; t < this.tokens; t++) for (let c = 0; c < FLUX2_CHANNELS; c++) z[c * this.tokens + t] = this.x[t * FLUX2_CHANNELS + c]!;
    const pre = await this.session('vaePre').run({
      z: new this.ort.Tensor('float16', toHalf(z), [1, FLUX2_CHANNELS, this.latentH, this.latentW]),
    });
    const residual = pre['/decoder/block_1/Add_output_0'] as Tensor;
    const q = pre['/decoder/attn_1/Mul_6_output_0'] as Tensor;
    const kt = pre['/decoder/attn_1/Mul_7_output_0'] as Tensor;
    const v = pre['/decoder/attn_1/Reshape_2_output_0'] as Tensor;
    // Attention over the decoder's middle (2 × the latent size, 512 channels), 1,024 query positions at a time.
    const seq = this.tokens * 4;
    const head = 512;
    const qData = q.data as Uint16Array;
    const attention = new Uint16Array(seq * head);
    for (let start = 0; start < seq; start += FLUX2_ATTENTION_CHUNK) {
      if (this.cancelled()) throw new RuntimeError('cancelled', 'Generation was stopped.');
      const n = Math.min(FLUX2_ATTENTION_CHUNK, seq - start);
      const chunk = await this.session('vaeAttention').run({
        q_chunk: new this.ort.Tensor('float16', halfBits(qData).slice(start * head, (start + n) * head), [1, 1, n, head]),
        kt,
        v,
      });
      attention.set(halfBits((chunk.attn_chunk as Tensor).data as Uint16Array), start * head);
      disposeExcept(chunk, []);
    }
    q.dispose();
    kt.dispose();
    v.dispose();
    let feeds: Record<string, Tensor> = {
      '/decoder/block_1/Add_output_0': residual,
      '/decoder/attn_1/MatMul_1_output_0': new this.ort.Tensor('float16', attention, [1, 1, seq, head]),
    };
    let image: Tensor | null = null;
    for (let i = 0; this.loaded.sessions.has(`vaePost${i}`); i++) {
      const session = this.session(`vaePost${i}`);
      const out = await session.run(feeds);
      for (const t of Object.values(feeds)) t.dispose();
      const name = session.outputNames[0]!;
      image = out[name] as Tensor;
      feeds = { [this.session(`vaePost${i + 1}`)?.inputNames[0] ?? 'done']: image };
    }
    const pixels = toRgba(readFloats(image!), this.width, this.height);
    image!.dispose();
    const canvas = new OffscreenCanvas(this.width, this.height);
    canvas.getContext('2d')!.putImageData(new ImageData(pixels, this.width, this.height), 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }

  dispose(): void {
    this.context?.dispose();
    this.context = null;
  }
}

/**
 * FLUX 3 Image Klein's generation (BFL's pipeline.py, text-to-image): flow-map steps over [tokens, 128] latents, one
 * token per 16 × 16 pixels; each step jumps from t to the next time (x += (t_ref − t) × pred). Image positions are
 * (0, row / 2, column / 2, 0), text positions (0, 0, 0, i). The decoder takes the tokens on their grid.
 */
class Flux3Run implements ImageRun {
  private readonly ort = getOrt();
  private readonly latentW: number;
  private readonly latentH: number;
  private readonly tokens: number;
  private x: Float32Array;
  private readonly schedule: number[];
  private context: { ctx: Tensor; ids: Float32Array; length: number } | null = null;

  constructor(
    private readonly loaded: Loaded,
    private readonly request: ImageGenerationRequest,
    readonly steps: number,
    seed: number,
    private readonly width: number,
    private readonly height: number,
    private readonly cancelled: () => boolean,
  ) {
    this.latentW = width / 16;
    this.latentH = height / 16;
    this.tokens = this.latentW * this.latentH;
    this.x = gaussian(this.tokens * FLUX2_CHANNELS, seed);
    this.schedule = flux3Schedule(steps);
  }

  private session(role: string): InferenceSession {
    return this.loaded.sessions.get(role)!;
  }

  async encodePrompt(): Promise<void> {
    const { ids, mask, positions } = flux3PromptIds(this.loaded.qwen!, this.request.prompt);
    const n = ids.length;
    const long = (values: number[]) => new this.ort.Tensor('int64', BigInt64Array.from(values, BigInt), [1, n]);
    const out = await this.session('textEncoder').run({ input_ids: long(ids), attention_mask: long(mask), position_ids: long(positions) });
    const textIds = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) textIds[i * 4 + 3] = i;
    this.context = { ctx: out.ctx as Tensor, ids: textIds, length: n };
  }

  async step(i: number): Promise<void> {
    const imageIds = new Float32Array(this.tokens * 4);
    for (let h = 0, k = 0; h < this.latentH; h++) for (let w = 0; w < this.latentW; w++, k += 4) imageIds.set([0, h / 2, w / 2, 0], k);
    const { ctx, ids, length } = this.context!;
    const t = this.schedule[i]!;
    const tRef = this.schedule[i + 1]!;
    const out = await this.session('transformer').run({
      x: new this.ort.Tensor('float32', this.x, [1, this.tokens, FLUX2_CHANNELS]),
      x_ids: new this.ort.Tensor('float32', imageIds, [1, this.tokens, 4]),
      t: new this.ort.Tensor('float32', Float32Array.of(t), [1]),
      t_ref: new this.ort.Tensor('float32', Float32Array.of(tRef), [1]),
      ctx,
      ctx_ids: new this.ort.Tensor('float32', ids, [1, length, 4]),
    });
    const pred = readFloats(out.pred as Tensor);
    disposeExcept(out, []);
    const dt = tRef - t;
    this.x = this.x.map((v, k) => v + dt * pred[k]!);
  }

  /** The decoder is too slow to run after every step; the image comes at the end. */
  async preview(): Promise<undefined> {
    return undefined;
  }

  async final(): Promise<Blob> {
    if (this.cancelled()) throw new RuntimeError('cancelled', 'Generation was stopped.');
    // [tokens, 128] in row order → [1, 128, h/16, w/16].
    const z = new Float32Array(this.x.length);
    for (let t = 0; t < this.tokens; t++) for (let c = 0; c < FLUX2_CHANNELS; c++) z[c * this.tokens + t] = this.x[t * FLUX2_CHANNELS + c]!;
    const out = await this.session('vaeDecoder').run({ z: new this.ort.Tensor('float32', z, [1, FLUX2_CHANNELS, this.latentH, this.latentW]) });
    const image = out.image as Tensor;
    const pixels = toRgba(readFloats(image), this.width, this.height);
    image.dispose();
    const canvas = new OffscreenCanvas(this.width, this.height);
    canvas.getContext('2d')!.putImageData(new ImageData(pixels, this.width, this.height), 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }

  dispose(): void {
    this.context?.ctx.dispose();
    this.context = null;
  }
}

/** float16 tensor data as raw half bits (it may come as a Float16Array). */
function halfBits(data: ArrayLike<number>): Uint16Array {
  return data instanceof Uint16Array ? data : toHalf(data);
}

/** KV tensor names of a language model's layers: `<prefix>.<i>.key` and `.value`. */
function kvNames(layers: number, prefix: 'past_key_values' | 'present'): string[] {
  return Array.from({ length: layers }, (_, i) => [`${prefix}.${i}.key`, `${prefix}.${i}.value`]).flat();
}

function tokenizers(pipeline: ImagePipeline, files: ImageFileSource): Pick<Loaded, 'clip' | 'clip2' | 'qwen' | 't5'> {
  const json = (file: ArtifactFile): unknown => JSON.parse(new TextDecoder().decode(files.bytes(file)));
  const clip = (t: ClipTokenizerFiles) =>
    t.json
      ? ClipTokenizer.fromJSON(json(t.json), t.pad)
      : new ClipTokenizer(json(t.vocab!) as Record<string, number>, new TextDecoder().decode(files.bytes(t.merges!)), t.pad);
  if (pipeline.kind === 'z-image' || pipeline.kind === 'janus' || pipeline.kind === 'flux2' || pipeline.kind === 'flux3')
    return { qwen: BpeTokenizer.fromJSON(json(pipeline.tokenizer)) };
  if (pipeline.kind === 'sdxl') return { clip: clip(pipeline.tokenizer), clip2: clip(pipeline.tokenizer2) };
  if (pipeline.kind === 'flux1') return { clip: clip(pipeline.tokenizer), t5: UnigramTokenizer.fromJSON(json(pipeline.tokenizer2)) };
  if (pipeline.kind === 'sd3')
    return { clip: clip(pipeline.tokenizer), clip2: clip(pipeline.tokenizer2), t5: UnigramTokenizer.fromJSON(json(pipeline.tokenizer3)) };
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
