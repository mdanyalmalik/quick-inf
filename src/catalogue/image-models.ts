// Image generation models. Each describes the inputs it takes beyond the prompt, so the workspace builds its
// settings from the model instead of assuming one set: a distilled model like SD Turbo takes a seed and a step or
// two, Stable Diffusion 1.5 also takes a negative prompt, a size and guidance. Models with `files` run in the app
// (src/runtime/image); the others are previews (placeholder images) until they do.

import { OWN_EXPORTS, type ArtifactFile, type Precision } from './models';

/** An input besides the prompt, shown under Settings. */
export type ImageInput =
  | { kind: 'text'; id: string; label: string; placeholder: string; hint?: string }
  | { kind: 'select'; id: string; label: string; options: { value: string; label: string }[]; default: string; hint?: string }
  | { kind: 'range'; id: string; label: string; min: number; max: number; step: number; default: number; hint?: string }
  | { kind: 'seed'; id: string; label: string; hint?: string };

/** One ONNX graph and the weight files beside it (named in the graph by their file names). */
export interface ImageGraph {
  graph: ArtifactFile;
  data?: ArtifactFile[];
  /** Overrides the graph's MatMulNBits accuracy_level. */
  matMulNBitsAccuracy?: number;
  /**
   * ONNX Runtime graph optimizations, when not 'all': FLUX.2 klein's export hand-patches its attention, which the
   * fusions break on WebGPU ("divide by zero"); its runtime creates every session with them off.
   */
  optimization?: 'disabled' | 'basic';
  /** Sizes for symbolic dimensions (as FLUX.2 klein's runtime fixes batch and text length). */
  freeDimensions?: Record<string, number>;
  /** Quantize the graph's float MatMul/Gemm weights to this many bits as it loads (see external-weights.ts). */
  quantizeWeights?: 4 | 8;
  /** Compute the float16 graph in float32 (Stable Diffusion 3.5's transformer overflows float16 above 512²). */
  computeFloat32?: boolean;
}

/** CLIP's tokenizer: vocab.json and merges.txt, or one tokenizer.json. */
export interface ClipTokenizerFiles {
  vocab?: ArtifactFile;
  merges?: ArtifactFile;
  json?: ArtifactFile;
  /** The token that pads prompts to 77: SD 2.x's (and SDXL's second) tokenizer pads with "!", SD 1.x's with <|endoftext|>. */
  pad: '!' | '<|endoftext|>';
}

/**
 * Euler over the model's noise levels. `spacing` picks the timesteps as diffusers' option of that name;
 * `ancestral` re-noises each step's predicted image with fresh noise (what adversarially distilled models like
 * SD Turbo are trained for; plain Euler over-sharpens them with every step).
 */
export interface EulerScheduler {
  kind: 'euler';
  spacing: 'trailing' | 'linspace' | 'leading';
  ancestral?: boolean;
}

/** TAESD (MIT): a 5 MB decoder for SD 1.x/2.x latents, ~1/20th of the full decoder's work. Raw latents in, [−1, 1] out. */
const TAESD: ArtifactFile = {
  path: 'taesd/vae_decoder.onnx',
  sizeBytes: 4912312,
  sha256: '0c9459fda7b8d582081be2494adbdaf806366728f49b8ba243f7dc132455ad10',
  from: { repo: 'julienkay/taesd', revision: 'c8a437fd0201c21c3bcd298fcf3181b063bcc1eb', path: 'vae_decoder/model.onnx' },
};

/** TAESDXL (MIT): the same for SDXL latents. */
const TAESDXL: ArtifactFile = {
  path: 'taesdxl/vae_decoder.onnx',
  sizeBytes: 4912312,
  sha256: 'a2911f6ecd3fb71bd5bc8656dce9d1bff402a20b81f59cf2b39450b810c2ca98',
  from: { repo: 'julienkay/taesdxl', revision: '4d9534fde1cf927367779db780b15aa108528d12', path: 'vae_decoder/model.onnx' },
};

/** How a model makes images (src/runtime/image/image-engine.ts runs each kind). */
export type ImagePipeline =
  | {
      /** Stable Diffusion 1.x/2.x and models built on it (SD Turbo, LCM Dreamshaper). */
      kind: 'sd';
      tokenizer: ClipTokenizerFiles;
      textEncoder: ImageGraph;
      unet: ImageGraph;
      vaeDecoder: ImageGraph;
      /** A tiny decoder (TAESD) for each step's preview; previews use the full decoder without one. */
      previewDecoder?: ImageGraph;
      /** Element type of the graphs' float inputs and outputs. */
      io: 'float32' | 'float16';
      timestep: 'int64' | 'float32';
      /** The UNet takes one timestep for the whole batch, as a scalar (Optimum's SD 1.5 exports). */
      scalarTimestep?: boolean;
      /** The text encoder's input_ids type (int32 by default). */
      textInput?: 'int32' | 'int64';
      /** The VAE decoder's input and output names, when not latent_sample → sample. */
      vaeNames?: { input: string; output: string };
      /** The VAE decoder divides by the latent scale (0.18215) itself. */
      vaeScales?: boolean;
      scheduler: EulerScheduler | { kind: 'lcm' };
      /**
       * 'cfg': the prompt and the negative prompt in one batch of 2, mixed by the guidance setting;
       * 'embedding': guidance is an input of the model (LCM). Without it the model is distilled to need none.
       */
      guidance?: 'cfg' | 'embedding';
    }
  | {
      /** SDXL (SDXL Turbo): two text encoders, size conditioning; float16 throughout. */
      kind: 'sdxl';
      tokenizer: ClipTokenizerFiles;
      tokenizer2: ClipTokenizerFiles;
      textEncoder: ImageGraph;
      textEncoder2: ImageGraph;
      unet: ImageGraph;
      vaeDecoder: ImageGraph;
      previewDecoder?: ImageGraph;
      scheduler: EulerScheduler;
      /** Float inputs and outputs: float16 (Microsoft's WebNN exports, the default) or float32 (Optimum exports). */
      io?: 'float16' | 'float32';
      /** The UNet's timestep type (float16 by default). */
      timestep?: 'float16' | 'float32' | 'int64';
      /** 'cfg': the prompt and the negative prompt as a batch of 2 (SDXL 1.0 and its finetunes). */
      guidance?: 'cfg';
    }
  | {
      /** Z-Image: a Qwen3 text encoder, a diffusion transformer and flow matching; float16 throughout. */
      kind: 'z-image';
      tokenizer: ArtifactFile;
      textEncoder: ImageGraph;
      transformer: ImageGraph;
      vaeDecoder: ImageGraph;
    }
  | {
      /**
       * FLUX.2 klein: Qwen3 text features (three layers, 512 tokens), a flow-matching transformer over 16 × 16-pixel
       * latent tokens, and a decoder split into stages (pre-attention, attention in chunks, post-attention).
       */
      kind: 'flux2';
      tokenizer: ArtifactFile;
      textEncoder: ImageGraph;
      transformer: ImageGraph;
      vaePre: ImageGraph;
      vaeAttention: ImageGraph;
      vaePost: ImageGraph[];
    }
  | {
      /**
       * Stable Diffusion 3.5: CLIP-L's and CLIP-G's penultimate layers side by side (padded to T5's width) followed by
       * T5-XXL's 256 tokens, and both CLIPs' pooled outputs; an MMDiT transformer over flow matching (shift 3);
       * a 16-channel VAE. Input types (token ids, float16 or float32) are read from the graphs.
       */
      kind: 'sd3';
      tokenizer: ClipTokenizerFiles;
      tokenizer2: ClipTokenizerFiles;
      /** T5's tokenizer.json (Unigram). */
      tokenizer3: ArtifactFile;
      textEncoder: ImageGraph;
      textEncoder2: ImageGraph;
      textEncoder3: ImageGraph;
      transformer: ImageGraph;
      vaeDecoder: ImageGraph;
      /** 'cfg': the prompt and the negative prompt as a batch of 2 (Medium, Large); Turbo needs none. */
      guidance?: 'cfg';
    }
  | {
      /**
       * FLUX.1 (schnell, dev): CLIP-L's pooled output and T5's tokens; a rectified-flow transformer over latents packed
       * into tokens of 2 × 2 × 16 values, with their positions as ids; a 16-channel VAE. schnell is distilled to need
       * neither guidance nor a shifted schedule; dev takes its guidance as an input and shifts the schedule by image size.
       */
      kind: 'flux1';
      tokenizer: ClipTokenizerFiles;
      /** T5's tokenizer.json (Unigram). */
      tokenizer2: ArtifactFile;
      textEncoder: ImageGraph;
      textEncoder2: ImageGraph;
      transformer: ImageGraph;
      vaeDecoder: ImageGraph;
      /** T5 tokens (diffusers: 256 for schnell, 512 for dev). */
      textTokens: number;
      /** dev: the schedule shifted for the image size (diffusers' dynamic shifting, 0.5 at 256 tokens to 1.15 at 4096). */
      shiftSchedule?: boolean;
      /** dev: the guidance setting is an input of the transformer (distilled guidance, one pass per step). */
      guidanceInput?: boolean;
    }
  | {
      /**
       * Janus: a language model that writes the image as 576 tokens (24 rows of 24) one at a time, then a decoder
       * turns them into pixels. Guidance runs the prompt and a blanked prompt side by side.
       */
      kind: 'janus';
      tokenizer: ArtifactFile;
      embedTokens: ImageGraph;
      languageModel: ImageGraph;
      genHead: ImageGraph;
      genImgEmbeds: ImageGraph;
      imageDecode: ImageGraph;
      layers: number;
    }
  | {
      /**
       * aMUSEd: a CLIP text encoder, a transformer that predicts every masked image token at once (its graph mixes the
       * prompt's and the negative prompt's predictions by the guidance setting), and a VQ decoder. All tokens start
       * masked; each step keeps the most confident of the new ones (see amusedStep).
       */
      kind: 'amused';
      tokenizer: ClipTokenizerFiles;
      textEncoder: ImageGraph;
      transformer: ImageGraph;
      vqDecoder: ImageGraph;
      /** The id that marks a position as still unknown. */
      maskToken: number;
    };

/** A model's licence, as the app must show it. */
export interface ModelLicense {
  name: string;
  url: string;
  /** Attribution the licence asks to be shown prominently. */
  notice?: string;
  /** The acceptable use policy the licence makes users follow. */
  usePolicyUrl?: string;
}

/** Stable Diffusion 1.x and models built on it: open, with use restrictions users must follow. */
const OPENRAIL_M: ModelLicense = {
  name: 'CreativeML OpenRAIL-M',
  url: 'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/LICENSE.md',
};

/** SDXL and models built on it. */
const OPENRAIL_PP: ModelLicense = {
  name: 'CreativeML OpenRAIL++-M',
  url: 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md',
};

const STABILITY_COMMUNITY: ModelLicense = {
  name: 'Stability AI Community License',
  url: 'https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md',
  notice: 'Powered by Stability AI',
  usePolicyUrl: 'https://stability.ai/use-policy',
};

/** Where `ImageModelSpec.arena` comes from. */
export const IMAGE_ARENA_SOURCE = 'Artificial Analysis text-to-image arena, Elo (artificialanalysis.ai, 13 Oct 2026)';

export interface ImageModelSpec {
  id: string;
  name: string;
  publisher: string;
  parameters: string;
  /** Total parameters, for ordering by size. */
  parameterCount: number;
  /**
   * Relative image quality among the image models (higher is better): prompt following, detail and how often
   * images come out right. Picks the recommended model among those that run well.
   */
  rank: number;
  /** Human preference: the model's Artificial Analysis text-to-image arena Elo (IMAGE_ARENA_SOURCE), when rated. */
  arena?: number;
  summary: string;
  /** Output size, as the overview shows it. */
  sizeLabel: string;
  inputs: ImageInput[];
  license?: ModelLicense;
  /** Where the files come from (pinned), for models that run. */
  source?: { host: 'huggingface'; repo: string; revision: string };
  /** Denoising steps when the model has no steps setting (SDXL Lightning is made for exactly 4). */
  steps?: number;
  /** Weight precision of the files, as the load panel shows it. */
  precision?: Precision;
  /** The pipeline and its files, for models that run; absent for previews. */
  pipeline?: ImagePipeline;
  /**
   * Times on a reference device and its device-test compute, for estimates elsewhere:
   * the text encoder and decoder (`fixedMs`) and each step with its preview (`perStepMs`).
   */
  reference?: { fixedMs: number; perStepMs: number; computeGflops: number };
  /** WebGPU features the files need (fp16 maths inside). */
  requiredGpuFeatures?: GPUFeatureName[];
}

/** Values of a model's inputs, by input id (seed included, -1 meaning a random seed each time). */
export type ImageSettings = Record<string, string | number>;

const SEED: ImageInput = { kind: 'seed', id: 'seed', label: 'Seed', hint: 'The same seed and settings give the same image.' };
const NEGATIVE: ImageInput = {
  kind: 'text',
  id: 'negative',
  label: 'Negative prompt',
  placeholder: 'What to avoid, e.g. blurry, text',
  hint: 'Things the image should not contain.',
};
const square = (sizes: number[]) => sizes.map((s) => ({ value: `${s}x${s}`, label: `${s} × ${s}` }));

export const IMAGE_MODELS: readonly ImageModelSpec[] = [
  // Under 1B parameters, small enough to run anywhere (2026-10-15): our own exports (tools/export/export_sd.py,
  // export_amused.py), float16 weights with float32 inputs and outputs.
  {
    id: 'sdxs-512-dreamshaper',
    name: 'SDXS DreamShaper',
    publisher: 'IDKiro',
    parameters: '440M',
    parameterCount: 440_000_000,
    rank: 1,
    summary: 'A Stable Diffusion cut down and trained to make a 512 × 512 image in a single step: about as fast as image generation gets.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_PP,
    precision: 'fp16',
    steps: 1,
    source: {
      host: 'huggingface',
      repo: `${OWN_EXPORTS}/sdxs-512-dreamshaper-ONNX-WebGPU`,
      revision: '3256e981453992aba867a15825f02a4988f2c791',
    },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 246323042,
          sha256: '03d29052ef18d87ebea938e212c5bed15c33e8afb55a681b802676756b438dc7',
        },
      },
      unet: {
        graph: {
          path: 'unet/model.onnx',
          sizeBytes: 631897533,
          sha256: '1af3c9644c9e0f50522ce46b6bcd77fb7d84429728eb4d1243a2ec966f2aa64f',
        },
      },
      // TAESD, the tiny decoder: latents as they are.
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 4912820,
          sha256: '801e913994dc73d9ec68643d555d4325a70b72f431ffbf33b2b1c278dd5da72f',
        },
      },
      io: 'float32',
      timestep: 'float32',
      vaeScales: true,
      scheduler: { kind: 'euler', spacing: 'trailing' },
    },
    reference: { fixedMs: 200, perStepMs: 220, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [SEED],
  },
  {
    id: 'tiny-sd',
    name: 'Tiny SD',
    publisher: 'Segmind',
    parameters: '530M',
    parameterCount: 530_000_000,
    rank: 0.5,
    summary:
      'Stable Diffusion 1.5 with half its blocks removed: the smallest model here that takes a negative prompt and guidance. Rough detail.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_M,
    precision: 'fp16',
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/tiny-sd-ONNX-WebGPU`, revision: 'bcb34e52bbb548d5d4bbb3b210c9fa1cd248a9ed' },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 246323042,
          sha256: 'a69bb1de280181343ad0920b41cf289e52ac70dbd6083f0ed0d6371fb9d2f29c',
        },
      },
      unet: {
        graph: {
          path: 'unet/model.onnx',
          sizeBytes: 647336426,
          sha256: '6bc30ad741a7de7c19f5ee4fa3c7298dfd6fa7308c16d144b5e321fe7810760f',
        },
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 198078716,
          sha256: '9883e574be18d9d616881e921eed5d5d22d0f8157b2872356f9536783dc8d86f',
        },
      },
      previewDecoder: { graph: TAESD },
      io: 'float32',
      timestep: 'float32',
      scheduler: { kind: 'euler', spacing: 'linspace' },
      guidance: 'cfg',
    },
    reference: { fixedMs: 2600, perStepMs: 860, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 25, hint: '20–30 is the sweet spot.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 7.5,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      SEED,
    ],
  },
  {
    id: 'bk-sdm-small-2m',
    name: 'BK-SDM Small',
    publisher: 'Nota AI',
    parameters: '690M',
    parameterCount: 690_000_000,
    rank: 1,
    summary: 'A compressed Stable Diffusion from Nota AI’s BK-SDM research (ECCV 2024), trained on 2M images: small and steady.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_M,
    precision: 'fp16',
    source: {
      host: 'huggingface',
      repo: `${OWN_EXPORTS}/bk-sdm-small-2m-ONNX-WebGPU`,
      revision: '96a7f8a4d398898d882960e67aa71c9961ec312c',
    },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 246323042,
          sha256: 'a50ad711372e4b795ff1ab3fa831990860b5f60228fb09a243377435703eb51f',
        },
      },
      unet: {
        graph: {
          path: 'unet/model.onnx',
          sizeBytes: 965284722,
          sha256: 'b620170e4270b4e37acc3a2a74d20c3972a5d66e5cb99864885e695168490bd9',
        },
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 198078716,
          sha256: '7f5b5c2337ca49e6c4fb6e227c80073511f8cc203724eeb8fc02bfe3446e8b72',
        },
      },
      previewDecoder: { graph: TAESD },
      io: 'float32',
      timestep: 'float32',
      scheduler: { kind: 'euler', spacing: 'leading' },
      guidance: 'cfg',
    },
    reference: { fixedMs: 2900, perStepMs: 1000, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 25, hint: '20–30 is the sweet spot.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 7.5,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      SEED,
    ],
  },
  {
    id: 'small-sd',
    name: 'Small SD',
    publisher: 'Segmind',
    parameters: '790M',
    parameterCount: 790_000_000,
    rank: 1,
    summary: 'Stable Diffusion 1.5 with a third of its blocks removed: noticeably sharper than Tiny SD, still under 1B parameters.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_M,
    precision: 'fp16',
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/small-sd-ONNX-WebGPU`, revision: 'c9a5872eb84c21fe5cd7c415dba2fc7989c30e98' },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 246323042,
          sha256: '4d40575f0cc05aa37263ff4100d415a034b3849fab6b1e9e872414e3989ccfe8',
        },
      },
      unet: {
        graph: {
          path: 'unet/model.onnx',
          sizeBytes: 1159426870,
          sha256: 'a56fc7035f861a7ec3f10252de2cc6881982611b8594bba6999a5fa383f6e056',
        },
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 198078716,
          sha256: '9883e574be18d9d616881e921eed5d5d22d0f8157b2872356f9536783dc8d86f',
        },
      },
      previewDecoder: { graph: TAESD },
      io: 'float32',
      timestep: 'float32',
      scheduler: { kind: 'euler', spacing: 'linspace' },
      guidance: 'cfg',
    },
    reference: { fixedMs: 3200, perStepMs: 1140, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 25, hint: '20–30 is the sweet spot.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 7.5,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      SEED,
    ],
  },
  {
    id: 'amused-512',
    name: 'aMUSEd',
    publisher: 'Hugging Face',
    parameters: '820M',
    parameterCount: 820_000_000,
    rank: 1,
    summary:
      'A different kind of model: it fills in a grid of image tokens over a dozen steps instead of removing noise. Small and quick; simple scenes.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_PP,
    precision: 'fp16',
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/amused-512-ONNX-WebGPU`, revision: '3363d27e637c77daf19f38af84a2b882f893c911' },
    pipeline: {
      kind: 'amused',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 247542596,
          sha256: 'f09f95d302bf9b8d3744b483c37a68ead421ef26537cba63c37452546267d630',
        },
      },
      transformer: {
        graph: {
          path: 'transformer/model.onnx',
          sizeBytes: 1217903919,
          sha256: '01ea871fec2c2e9388041cc1cefb145e69cdd8838dca3a37a06f8cda26dafcfc',
        },
      },
      vqDecoder: {
        graph: {
          path: 'vq_decoder/model.onnx',
          sizeBytes: 350307847,
          sha256: 'c41279403b23bd6aaac521e28676ddb0850965cc3688fb9200ce64ac8e44b58e',
        },
      },
      maskToken: 8255,
    },
    reference: { fixedMs: 1900, perStepMs: 2070, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 8, max: 24, step: 1, default: 12, hint: '12 is the model’s default.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 10,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      SEED,
    ],
  },

  {
    id: 'sd-turbo',
    name: 'SD Turbo',
    publisher: 'Stability AI',
    parameters: '1B',
    parameterCount: 1_000_000_000,
    rank: 1,
    summary: 'A distilled Stable Diffusion: a usable image in a single step.',
    sizeLabel: '512 × 512',
    // Stability moved SD Turbo to its Community License on 2024-07-05 (this export predates that and still carries the
    // old non-commercial file; the weights are Stability's, unchanged apart from the format).
    license: STABILITY_COMMUNITY,
    // fp16 ONNX export made for ONNX Runtime Web (the one Microsoft's WebGPU demo uses).
    source: { host: 'huggingface', repo: 'schmuell/sd-turbo-ort-web', revision: 'ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1' },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 681393168,
          sha256: '5d1f5bd3286abf26313a9e87db6fd32e47d8b1b2a84ffdf470884c3fe89497c4',
        },
      },
      unet: {
        graph: {
          path: 'unet/model.onnx',
          sizeBytes: 1733430199,
          sha256: '7468133d9f7babec9cf6636e160188fe25aad206792c0abdaa722edc676963c4',
        },
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 99094314,
          sha256: 'b37a85f91a982811957abd6cfb617499742309f7b8e253240da3132e533a7605',
        },
      },
      previewDecoder: { graph: TAESD },
      io: 'float32',
      timestep: 'int64',
      scheduler: { kind: 'euler', spacing: 'trailing', ancestral: true },
    },
    reference: { fixedMs: 1790, perStepMs: 850, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      {
        kind: 'range',
        id: 'steps',
        label: 'Steps',
        min: 1,
        max: 4,
        step: 1,
        default: 1,
        hint: '1 step is fastest; up to 4 add detail and realism.',
      },
      SEED,
    ],
  },
  {
    id: 'lcm-dreamshaper-v7',
    name: 'LCM Dreamshaper v7',
    publisher: 'Tsinghua (LCM) · Lykon',
    parameters: '1B',
    parameterCount: 1_066_000_000,
    rank: 2,
    summary: 'Dreamshaper, a popular artistic Stable Diffusion 1.5 model, distilled to make an image in four steps.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_M,
    precision: 'fp16',
    source: { host: 'huggingface', repo: 'aislamov/lcm-dreamshaper-v7-onnx', revision: '7dbd8a0cd488d9a66ec3505322269a827fccfe0d' },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 246519878,
          sha256: '5475b0bca29a8a7f1153f10273ed24c359a4bf54463024a22dd3bc9ed447698e',
        },
      },
      unet: {
        graph: { path: 'unet/model.onnx', sizeBytes: 93694515, sha256: '7cb6412679efb6a4316e249847cffd8ed6ca8d3937c8e602a4c41271e9a0a335' },
        data: [
          {
            path: 'unet/model.onnx_data',
            sizeBytes: 1644797440,
            sha256: '0834090de51aa5eab01a9e0755df0ce0327389580028365610bed63005ff543d',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 2111331,
          sha256: '19718d5250b6c4e133f456ceb278cc7d9c72e56dd545aa90e5a39269e05675ee',
        },
        data: [
          {
            path: 'vae_decoder/model.onnx_data',
            sizeBytes: 97389312,
            sha256: 'ea34e066517f641bf43880cada728182d4110a37cf6b5cde47e23370fb16fbc8',
          },
        ],
      },
      previewDecoder: { graph: TAESD },
      io: 'float32',
      timestep: 'float32',
      scheduler: { kind: 'lcm' },
      guidance: 'embedding',
    },
    reference: { fixedMs: 1890, perStepMs: 660, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      {
        kind: 'range',
        id: 'steps',
        label: 'Steps',
        min: 1,
        max: 8,
        step: 1,
        default: 4,
        hint: '4 is the sweet spot; 2 is quicker and softer.',
      },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 8,
        hint: 'How closely to follow the prompt.',
      },
      SEED,
    ],
  },
  {
    id: 'dreamshaper-8-lcm',
    name: 'DreamShaper 8 LCM',
    publisher: 'Lykon',
    parameters: '1B',
    parameterCount: 1_066_000_000,
    rank: 3,
    summary: 'DreamShaper 8, a much-loved all-round Stable Diffusion 1.5 model, distilled to make an image in four steps.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_M,
    precision: 'fp16',
    // fp16 inside, float32 inputs and outputs; the decoder applies the latent scale itself.
    source: {
      host: 'huggingface',
      repo: 'skillsafeai/dreamshaper-8-lcm-onnx-webgpu',
      revision: 'a310400909a293a7a937a54854910ed06469192b',
    },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder.onnx',
          sizeBytes: 246275486,
          sha256: '43f7a91340052d59c370b842388a34d5c337fe7323b268ba8e7b55795a1ec6f5',
        },
      },
      unet: {
        graph: { path: 'unet.onnx', sizeBytes: 1719739393, sha256: '1851819e721a81cc7498c932ca040fb1c2032c6a82e5f24c2c87d283ad3c0e41' },
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder.onnx',
          sizeBytes: 198068653,
          sha256: '08d65bbf93d7ecbb3b4765d713ee10d4623d4932a8006b2837b85909a86406ef',
        },
      },
      previewDecoder: { graph: TAESD },
      io: 'float32',
      timestep: 'float32',
      textInput: 'int64',
      vaeNames: { input: 'latent', output: 'image' },
      vaeScales: true,
      scheduler: { kind: 'lcm' },
    },
    reference: { fixedMs: 2350, perStepMs: 665, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      {
        kind: 'range',
        id: 'steps',
        label: 'Steps',
        min: 2,
        max: 8,
        step: 1,
        default: 4,
        hint: '4 is the sweet spot; 2 is quicker and softer.',
      },
      SEED,
    ],
  },
  {
    id: 'realistic-vision-6',
    name: 'Realistic Vision 6',
    publisher: 'SG161222',
    parameters: '1B',
    parameterCount: 1_066_000_000,
    rank: 4,
    summary:
      'The best-known photorealistic Stable Diffusion 1.5 model, especially for people and portraits. 25 steps with a negative prompt.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_M,
    precision: 'fp32',
    // Realistic Vision V6.0 B1 with the sd-vae-ft-mse decoder, float32 (an Optimum export).
    source: { host: 'huggingface', repo: 'Heliosoph/realistic-vision-cfg-onnx', revision: '20f2bfe722fa883f7852578f3335af1d437bcc75' },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 492588311,
          sha256: '430791d3b31e319ab59d04a6b770ad4bae5513fb7569effdc82707bc3c2cc650',
        },
      },
      unet: {
        graph: { path: 'unet/model.onnx', sizeBytes: 1206821, sha256: '21d87eb51da5c6de1779e4f6a1d2928a87cce34b15d5fab915570bf2d193c1c3' },
        data: [
          {
            path: 'unet/model.onnx_data',
            sizeBytes: 3438083840,
            sha256: 'c909f1831ba3fec5fdbd3a3e967b0700484470c647475930edf0ee902a8dc5c0',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 198078175,
          sha256: '20bf85ad399aafeac9ce47409f6a72f0d5ddb1becc2488ef4460a594fae2b752',
        },
      },
      previewDecoder: { graph: TAESD },
      io: 'float32',
      timestep: 'float32',
      scalarTimestep: true,
      textInput: 'int64',
      scheduler: { kind: 'euler', spacing: 'leading' },
      guidance: 'cfg',
    },
    reference: { fixedMs: 4000, perStepMs: 3000, computeGflops: 809 },
    requiredGpuFeatures: [],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 25, hint: '20–30 is the sweet spot.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 7.5,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      SEED,
    ],
  },
  {
    id: 'sdxl-lightning',
    name: 'SDXL Lightning',
    publisher: 'ByteDance',
    parameters: '3.5B',
    parameterCount: 3_500_000_000,
    rank: 6,
    arena: 720,
    summary: 'SDXL distilled to make a detailed 1024 × 1024 image in four steps.',
    sizeLabel: '1024 × 1024',
    license: OPENRAIL_PP,
    precision: 'fp16',
    steps: 4,
    // fp16 weights and maths, float32 inputs and outputs: d0gr's WebGPU export (text encoders in fp16 too).
    source: { host: 'huggingface', repo: 'd0gr/sdxl-lightning-onnx-webgpu', revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374' },
    pipeline: {
      kind: 'sdxl',
      tokenizer: {
        json: {
          path: 'tokenizer/tokenizer.json',
          sizeBytes: 1681537,
          sha256: '35bd8b310ba98bf39131d2f0907273a796e7c56bb824b34cb8287a170a77c717',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        json: {
          path: 'tokenizer_2/tokenizer.json',
          sizeBytes: 1681644,
          sha256: 'bba43011e86d372b3f706213bacfbe1fbf59421c44be801300479bfd07297f9a',
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 360437,
          sha256: '21267b285b3e3ab33fc6d95610ce629a67e0bca01647c36e65e6cd04b6eaa6e7',
        },
        data: [
          {
            path: 'text_encoder/model.onnx_data',
            sizeBytes: 246120960,
            sha256: '4b61db0f4781c335e6bba50ab7471bbfe02aad41d050ff5935dcbb2d7724fda0',
          },
        ],
      },
      textEncoder2: {
        graph: {
          path: 'text_encoder_2/model.onnx',
          sizeBytes: 1047642,
          sha256: '0cf6265108359805f27b16141209c83b9452d4ac66f7b626c6c6f046f648f484',
        },
        data: [
          {
            path: 'text_encoder_2/model.onnx_data',
            sizeBytes: 1389319680,
            sha256: '81330f8daecb663019d641658449d3d76205d90c2207356b9cccf1a427214c12',
          },
        ],
      },
      unet: {
        graph: { path: 'unet/model.onnx', sizeBytes: 4061552, sha256: 'cc0841c261fdcd4aeb0af0582e5e7726f63edcd67515adc9508f985566001698' },
        data: [
          {
            path: 'unet/model.onnx_data',
            sizeBytes: 2000000000,
            sha256: '99bf38881b2491d3d78789c9e7bbb1a81bdada86c143040417e69676700d64de',
          },
          {
            path: 'unet/model.onnx_data_1',
            sizeBytes: 2000000000,
            sha256: '2e9e7629e2fc2a4c4d4afb5aa7779e3c50ee17e3612f26af07f6019b61eff56d',
          },
          {
            path: 'unet/model.onnx_data_2',
            sizeBytes: 1134903040,
            sha256: 'e50c0ee78767cfbdc49c905fc441d4736f40cb9c679ebf6252ecf3c392181036',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 135522,
          sha256: 'd2dbbf47fba8ee8b1ec26c10b8a259d6e9870ec7d921df2b2d473fb97a2d0c03',
        },
        data: [
          {
            path: 'vae_decoder/model.onnx_data',
            sizeBytes: 98965248,
            sha256: 'fc3ff1700fd5f86a3e789aaadf647aa7742f2c739be6221ca692c5c92a21d39c',
          },
        ],
      },
      previewDecoder: { graph: TAESDXL },
      scheduler: { kind: 'euler', spacing: 'trailing' },
      io: 'float32',
      timestep: 'int64',
    },
    reference: { fixedMs: 7000, perStepMs: 4500, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [SEED],
  },
  {
    id: 'stable-diffusion-1.5',
    name: 'Stable Diffusion 1.5',
    publisher: 'Runway · Stability AI',
    parameters: '1B',
    parameterCount: 1_060_000_000,
    rank: 2,
    arena: 456,
    summary: 'The classic open image model: slower (25 steps, with a negative prompt), flexible, with many styles.',
    sizeLabel: '512 × 512',
    license: OPENRAIL_M,
    precision: 'fp16',
    // Microsoft's float16 export for its WebNN demo (the UNet takes the prompt and negative prompt as one batch of 2).
    source: { host: 'huggingface', repo: 'microsoft/stable-diffusion-v1.5-webnn', revision: 'd7c229b23e94f49b4a74c279ba17ff1afb367c31' },
    pipeline: {
      kind: 'sd',
      tokenizer: {
        vocab: {
          path: 'tokenizer/resolve/main/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/resolve/main/merges.txt',
          sizeBytes: 524618,
          sha256: 'f1cd414c2d113a28b296afc0fd66ae854c0e0c34d5de373a2f5b7f62e0c02344',
        },
        pad: '<|endoftext|>',
      },
      textEncoder: {
        graph: {
          path: 'Stable-Diffusion-v1.5-text-encoder-float16.onnx',
          sizeBytes: 246461779,
          sha256: '52a100df352408d691cd741f9887827cb1e211608cad3ff4f9d85b22127eafee',
        },
      },
      unet: {
        graph: {
          path: 'sd-unet-v1.5-model-b2c4h64w64s77-float16-compute-and-inputs-layernorm.onnx',
          sizeBytes: 1719405068,
          sha256: 'd680d78f36f5722c0205258337e91c41e0d552d17f812cffbe4f3fb138de3c82',
        },
      },
      vaeDecoder: {
        graph: {
          path: 'Stable-Diffusion-v1.5-vae-decoder-float16-fp32-instancenorm.onnx',
          sizeBytes: 99124333,
          sha256: 'b25ae0f7ae9877104fe9bf6e9ca53a0b07562de741c4cd86f16e5205d56b3b6f',
        },
      },
      previewDecoder: { graph: TAESD },
      io: 'float16',
      timestep: 'int64',
      scheduler: { kind: 'euler', spacing: 'linspace' },
      guidance: 'cfg',
    },
    reference: { fixedMs: 2150, perStepMs: 1460, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 25, hint: 'More steps: more detail, slower.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 7.5,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      SEED,
    ],
  },
  {
    id: 'sdxl-turbo',
    name: 'SDXL Turbo',
    publisher: 'Stability AI',
    parameters: '3.5B',
    parameterCount: 3_500_000_000,
    rank: 4,
    summary: 'A distilled SDXL: sharper, more detailed images than SD Turbo in one to four steps.',
    sizeLabel: '512 × 512',
    license: { ...STABILITY_COMMUNITY, url: 'https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md' },
    precision: 'q4f16',
    // 4-bit weights, float16 maths: Microsoft's export for its WebNN demo.
    source: { host: 'huggingface', repo: 'webnn/sdxl-turbo', revision: '51606a881f73accd9e69e3638c5b160a76d0eec3' },
    pipeline: {
      kind: 'sdxl',
      tokenizer: {
        json: {
          path: 'tokenizer/tokenizer.json',
          sizeBytes: 2322504,
          sha256: '577ee5f1ca023c15f72ba52cc9884dad27bbb1e5fbda0e2bf8fab406dccf01ad',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        json: {
          path: 'tokenizer_2/tokenizer.json',
          sizeBytes: 2322686,
          sha256: 'fdc32a56e63b4dbf8a4854e180e5714de0257e8b1e94e96c635e0894f49968c6',
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'onnx/text_encoder_model_q4f16.onnx',
          sizeBytes: 124433929,
          sha256: '329566cc7e406df0264d3fa936197152b25b39f2e9cf4b509daa3288e2b78666',
        },
      },
      textEncoder2: {
        graph: {
          path: 'onnx/text_encoder_2_model_q4f16.onnx',
          sizeBytes: 483677630,
          sha256: 'b95ae46553eb8136320e1a6410d5ab69526c51772b1df55c9921ab8b13c9b9e9',
        },
      },
      unet: {
        graph: {
          path: 'onnx/unet_model_q4f16.onnx',
          sizeBytes: 1965607674,
          sha256: 'eb2b07ef7b77d1e40ddffa8328e918637f543308e67d8c74cf0819f582370499',
        },
      },
      vaeDecoder: {
        graph: {
          path: 'onnx/vae_decoder_model_q4f16.onnx',
          sizeBytes: 97617257,
          sha256: '8a89610a245c01978c5a730dc749e6e3afb670a9828438c9323923a0c1f9f1ef',
        },
      },
      previewDecoder: { graph: TAESDXL },
      scheduler: { kind: 'euler', spacing: 'trailing', ancestral: true },
    },
    reference: { fixedMs: 1775, perStepMs: 1050, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      { kind: 'range', id: 'steps', label: 'Steps', min: 1, max: 4, step: 1, default: 1, hint: '1 step is fastest; up to 4 add detail.' },
      SEED,
    ],
  },
  {
    id: 'realvisxl-5-lightning',
    name: 'RealVisXL 5 Lightning',
    publisher: 'SG161222',
    parameters: '3.5B',
    parameterCount: 3_500_000_000,
    rank: 7,
    summary: 'A popular photorealistic SDXL model, distilled for fast images: detailed people, places and products at 1024 × 1024.',
    sizeLabel: '1024 × 1024',
    license: OPENRAIL_PP,
    precision: 'fp16',
    // d0gr's WebGPU export of the UNet; the text encoders and decoder are stock SDXL's (as RealVisXL's are, within
    // fp16 noise), shared from d0gr's SDXL Lightning repo.
    source: { host: 'huggingface', repo: 'd0gr/realvisxl-v5-lightning-unet-onnx', revision: '963e32e4f743f494bf4babf976342815977ad7c9' },
    pipeline: {
      kind: 'sdxl',
      tokenizer: {
        json: {
          path: 'sdxl/tokenizer/tokenizer.json',
          sizeBytes: 1681537,
          sha256: '35bd8b310ba98bf39131d2f0907273a796e7c56bb824b34cb8287a170a77c717',
          from: {
            repo: 'd0gr/sdxl-lightning-onnx-webgpu',
            revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
            path: 'tokenizer/tokenizer.json',
          },
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        json: {
          path: 'sdxl/tokenizer_2/tokenizer.json',
          sizeBytes: 1681644,
          sha256: 'bba43011e86d372b3f706213bacfbe1fbf59421c44be801300479bfd07297f9a',
          from: {
            repo: 'd0gr/sdxl-lightning-onnx-webgpu',
            revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
            path: 'tokenizer_2/tokenizer.json',
          },
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'sdxl/text_encoder/model.onnx',
          sizeBytes: 360437,
          sha256: '21267b285b3e3ab33fc6d95610ce629a67e0bca01647c36e65e6cd04b6eaa6e7',
          from: {
            repo: 'd0gr/sdxl-lightning-onnx-webgpu',
            revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
            path: 'text_encoder/model.onnx',
          },
        },
        data: [
          {
            path: 'sdxl/text_encoder/model.onnx_data',
            sizeBytes: 246120960,
            sha256: '4b61db0f4781c335e6bba50ab7471bbfe02aad41d050ff5935dcbb2d7724fda0',
            from: {
              repo: 'd0gr/sdxl-lightning-onnx-webgpu',
              revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
              path: 'text_encoder/model.onnx_data',
            },
          },
        ],
      },
      textEncoder2: {
        graph: {
          path: 'sdxl/text_encoder_2/model.onnx',
          sizeBytes: 1047642,
          sha256: '0cf6265108359805f27b16141209c83b9452d4ac66f7b626c6c6f046f648f484',
          from: {
            repo: 'd0gr/sdxl-lightning-onnx-webgpu',
            revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
            path: 'text_encoder_2/model.onnx',
          },
        },
        data: [
          {
            path: 'sdxl/text_encoder_2/model.onnx_data',
            sizeBytes: 1389319680,
            sha256: '81330f8daecb663019d641658449d3d76205d90c2207356b9cccf1a427214c12',
            from: {
              repo: 'd0gr/sdxl-lightning-onnx-webgpu',
              revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
              path: 'text_encoder_2/model.onnx_data',
            },
          },
        ],
      },
      unet: {
        graph: {
          path: 'unet/model_io32.onnx',
          sizeBytes: 3680343,
          sha256: 'c267cb941df83482b08feda6a749d958e2d84ac3581f348e569335319a75d246',
        },
        data: [
          {
            path: 'unet/model.onnx_data',
            sizeBytes: 1884519680,
            sha256: '0fd25b5651e94b408635e1bc18b15b9e93cb90769a96cdc9b4a5ac642479ba2b',
          },
          {
            path: 'unet/model.onnx_data_1',
            sizeBytes: 1895367680,
            sha256: '69454436dae4f29dd0df2ace899bb16975654fc909f502fdddf3bea4acd61f27',
          },
          {
            path: 'unet/model.onnx_data_2',
            sizeBytes: 1355015680,
            sha256: '281b26ba4f23f4c845c258d420be00de389d9fa1740927ba2d8e3797a01f211e',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'sdxl/vae_decoder/model.onnx',
          sizeBytes: 135522,
          sha256: 'd2dbbf47fba8ee8b1ec26c10b8a259d6e9870ec7d921df2b2d473fb97a2d0c03',
          from: {
            repo: 'd0gr/sdxl-lightning-onnx-webgpu',
            revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
            path: 'vae_decoder/model.onnx',
          },
        },
        data: [
          {
            path: 'sdxl/vae_decoder/model.onnx_data',
            sizeBytes: 98965248,
            sha256: 'fc3ff1700fd5f86a3e789aaadf647aa7742f2c739be6221ca692c5c92a21d39c',
            from: {
              repo: 'd0gr/sdxl-lightning-onnx-webgpu',
              revision: 'c0129962aa1a3f8ea02b3d55ddce5b3282342374',
              path: 'vae_decoder/model.onnx_data',
            },
          },
        ],
      },
      previewDecoder: { graph: TAESDXL },
      scheduler: { kind: 'euler', spacing: 'trailing' },
      io: 'float32',
      timestep: 'int64',
    },
    reference: { fixedMs: 8500, perStepMs: 5200, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [{ kind: 'range', id: 'steps', label: 'Steps', min: 4, max: 8, step: 1, default: 5, hint: 'It’s made for 4–6 steps.' }, SEED],
  },
  {
    id: 'z-image-turbo',
    name: 'Z-Image Turbo',
    publisher: 'Alibaba (Tongyi)',
    parameters: '6B',
    parameterCount: 6_150_000_000,
    rank: 9,
    arena: 941,
    summary: 'A large, recent model with excellent realism and prompt following, even for text in images. Needs a strong GPU.',
    sizeLabel: '512 × 512 or 768 × 768',
    license: { name: 'Apache 2.0', url: 'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo' },
    precision: 'q4f16',
    // 4-bit weights, float16 maths: Microsoft and Intel's export for the WebNN demo. The text encoder is Qwen3 4B.
    source: { host: 'huggingface', repo: 'webnn/Z-Image-Turbo', revision: '02e1d8cc5d10d9ff6c7ed1f54b173bd981381bc2' },
    pipeline: {
      kind: 'z-image',
      tokenizer: {
        path: 'tokenizer/tokenizer.json',
        sizeBytes: 11422654,
        sha256: 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
      },
      textEncoder: {
        graph: {
          path: 'onnx/text_encoder_model_q4f16.onnx',
          sizeBytes: 57983622,
          sha256: '97cb87b20147da5ba44c3666ec17b7d9a1c90cabbaee71649bcb1a374371fb66',
        },
        data: [
          {
            path: 'onnx/text_encoder_model_q4f16.onnx_data',
            sizeBytes: 2036236288,
            sha256: '5a23fd9188f4ac8eb5e0ae3cb9eae970005f869eac499c7048d951db30a04617',
          },
          {
            path: 'onnx/text_encoder_model_q4f16.onnx_data_1',
            sizeBytes: 124387328,
            sha256: '5c10bbb8670f2605f3bbc11fd302b15c4fdf264a2ac132d4ebec17717196cf6e',
          },
        ],
      },
      transformer: {
        graph: {
          path: 'onnx/transformer_model_q4f16.onnx',
          sizeBytes: 129947430,
          sha256: 'a19a13c20e040041dd5298bc57e24158fd4bae6ef1127beb26947c23b42ec7ff',
        },
        data: [
          {
            path: 'onnx/transformer_model_q4f16.onnx_data',
            sizeBytes: 2032271360,
            sha256: '19889aa5bce49b5b79b6c736d276797a6e826a7f3cb51f3e71b2d416cbbdebc7',
          },
          {
            path: 'onnx/transformer_model_q4f16.onnx_data_1',
            sizeBytes: 1507590144,
            sha256: '562bfd080c3ebbb5aaf176e47b24a71003403a4e0f9c5344ac85f8e687da1e47',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'onnx/vae_decoder_model_f16.onnx',
          sizeBytes: 99209498,
          sha256: '7b194f594d893ebcdd9afc6cf06944105fc58493870eb810ee7d4d5c333d8544',
        },
      },
    },
    reference: { fixedMs: 2050, perStepMs: 7300, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([512, 768]),
        default: '512x512',
        hint: '768 × 768 takes about twice as long.',
      },
      {
        kind: 'range',
        id: 'steps',
        label: 'Steps',
        min: 4,
        max: 12,
        step: 1,
        default: 9,
        hint: 'It’s tuned for 9; fewer is quicker but rougher.',
      },
      SEED,
    ],
  },
  {
    id: 'janus-pro-1b',
    name: 'Janus Pro 1B',
    publisher: 'DeepSeek',
    parameters: '1.5B',
    parameterCount: 1_500_000_000,
    rank: 3,
    summary: 'A language model that also draws, row by row instead of by removing noise; follows detailed prompts well.',
    sizeLabel: '384 × 384',
    license: { name: 'DeepSeek Model License', url: 'https://huggingface.co/deepseek-ai/Janus-Pro-1B/blob/main/LICENSE' },
    precision: 'q4f16',
    source: { host: 'huggingface', repo: 'onnx-community/Janus-Pro-1B-ONNX', revision: '04efdf2e36cb07a034b0d94f7322356b292f0418' },
    pipeline: {
      kind: 'janus',
      tokenizer: { path: 'tokenizer.json', sizeBytes: 4718995, sha256: 'cd187abdb6927a04ff0214b2d3fa05f4131de66298c8b3b75fbc25ce5f7faff2' },
      embedTokens: {
        graph: {
          path: 'onnx/embed_tokens_q4f16.onnx',
          sizeBytes: 419430808,
          sha256: '1fb1bca15de5024f723485b8e09fcea6d5d3eea1a6ffa9ccf11883259237748d',
        },
      },
      languageModel: {
        graph: {
          path: 'onnx/language_model_q4f16.onnx',
          sizeBytes: 698231365,
          sha256: 'bda78684bd85d4beda5558179e2a5da960f13969f111a8024f75be11cdef9a61',
        },
      },
      genHead: {
        graph: {
          path: 'onnx/gen_head_fp16.onnx',
          sizeBytes: 75536078,
          sha256: '59563c2c872555def6959653ad078bfe93d3484f6f9e692e25160ac2dadf4eae',
        },
      },
      genImgEmbeds: {
        graph: {
          path: 'onnx/gen_img_embeds_fp16.onnx',
          sizeBytes: 8695058,
          sha256: '1501cac82ce8d4b26b27af5f9f993b1575e27a42a09d07ac508feaebb79aedde',
        },
      },
      imageDecode: {
        graph: {
          path: 'onnx/image_decode_fp16.onnx',
          sizeBytes: 85297169,
          sha256: '2c29001e88eb560f2fa17736784fb33e15ba6933612321463d7818234eb84a09',
        },
      },
      layers: 24,
    },
    reference: { fixedMs: 20200, perStepMs: 0, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 10,
        step: 0.5,
        default: 5,
        hint: 'How closely to follow the prompt.',
      },
      SEED,
    ],
  },
  {
    id: 'janus-pro-7b',
    name: 'Janus Pro 7B',
    publisher: 'DeepSeek',
    parameters: '7B',
    parameterCount: 7_000_000_000,
    rank: 6,
    arena: 533,
    summary:
      'DeepSeek’s larger language model that also draws: follows detailed prompts closely, row by row instead of by removing noise. 384 × 384, a 4.6 GB download.',
    sizeLabel: '384 × 384',
    license: { name: 'DeepSeek Model License', url: 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/blob/main/LICENSE' },
    precision: 'q4f16',
    // Our own export (tools/export/export_dense.py janus-pro-7b + export_janus.py): the image-generation part only.
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Janus-Pro-7B-ONNX-WebGPU`, revision: 'dd7b2b1de3f3b2eb9f8f50c0160a1f55176a9936' },
    pipeline: {
      kind: 'janus',
      tokenizer: { path: 'tokenizer.json', sizeBytes: 7614107, sha256: '42bcf2c54739affa70425520f9f8eb48e7409cf515541c74594de4c412b7d5ad' },
      embedTokens: {
        graph: {
          path: 'embed_tokens.onnx',
          sizeBytes: 838861034,
          sha256: '34c5a8c09b4fc508e338ec1a4a374893d47eeb98c66d10a49abb14cdbe5e5ab1',
        },
      },
      languageModel: {
        graph: {
          path: 'language_model.onnx',
          sizeBytes: 202039,
          sha256: 'cb7406b9f58ab40c368ca3c5b540b333dbb4d2908df7f582f9d2be1bd808145e',
        },
        data: [
          {
            path: 'language_model.onnx.data',
            sizeBytes: 3419774976,
            sha256: '40181250e85d86602035a065356a4b6679ff845885275da141f4f6f5870bff4d',
          },
        ],
      },
      genHead: {
        graph: { path: 'gen_head.onnx', sizeBytes: 167814569, sha256: '507939ce67f118e3072581830f6214f0742bc9e2f887c1a7c02c226129e1081f' },
      },
      genImgEmbeds: {
        graph: {
          path: 'gen_img_embeds.onnx',
          sizeBytes: 33900392,
          sha256: 'ecedfbd347871c448c32e0f9b4aff4b1c66f3f749fe083d80fc560cc1a0120fe',
        },
      },
      imageDecode: {
        graph: {
          path: 'image_decode.onnx',
          sizeBytes: 170445942,
          sha256: '22726ffa871480308375d6f5d1e9ba04ba92af848ba6b9a96f0bb4b41c93723d',
        },
      },
      layers: 30,
    },
    reference: { fixedMs: 57400, perStepMs: 0, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 10,
        step: 0.5,
        default: 5,
        hint: 'How closely to follow the prompt.',
      },
      SEED,
    ],
  },
  {
    id: 'sdxl-1.0',
    name: 'SDXL 1.0',
    publisher: 'Stability AI',
    parameters: '3.5B',
    parameterCount: 3_500_000_000,
    rank: 7,
    arena: 678,
    summary:
      'The most widely used open image model: versatile and detailed at 1024 × 1024, with a negative prompt. About 25 steps, so slower than the distilled models.',
    sizeLabel: '1024 × 1024',
    license: OPENRAIL_PP,
    precision: 'fp16',
    // AMD's float16 GPU export (Olive): its GroupNorm nodes are rewritten as standard ops at load (external-weights.ts); the second text encoder is 4-bit.
    source: { host: 'huggingface', repo: 'amd/stable-diffusion-xl-1.0_io16_amdgpu', revision: '568216a64391f81fdb87947c1c549a2158db9224' },
    pipeline: {
      kind: 'sdxl',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        vocab: {
          path: 'tokenizer_2/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'tokenizer_2/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 246473185,
          sha256: 'adf384c0dc8e8404daf64d0aaf87fd39824f32519d0eaea81d86bbfa136bbad6',
        },
      },
      textEncoder2: {
        graph: {
          path: 'text_encoder_2/model.onnx',
          sizeBytes: 1050204,
          sha256: 'f2b2244d5f2e6911f8f85fa8a7056cf71982d81e4663d398f494cdf1389efb21',
        },
        data: [
          {
            path: 'text_encoder_2/model.onnx.data',
            sizeBytes: 452999680,
            sha256: 'ee00202b1fd802f8f0d4f2dfa3e6a44dd977dc39e3e28e1d40473dbb87aa644e',
          },
        ],
      },
      unet: {
        graph: { path: 'unet/model.onnx', sizeBytes: 1402843, sha256: 'a43d2aa66a74e270154d2abc9e98929d0da3de37c0ae9c1c60d8896a252d2ac4' },
        data: [
          {
            path: 'unet/model.onnx.data',
            sizeBytes: 5134903040,
            sha256: 'fc8c91ebd576bbef56a50166acd0bfc852abe4a6b91617ce91901c0c301a42f9',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 102364387,
          sha256: 'a80b7cfdcb4eb74f6598845e6a9f93e619df962f448c782a7789204cc03cdaa2',
        },
      },
      previewDecoder: { graph: TAESDXL },
      scheduler: { kind: 'euler', spacing: 'leading' },
      io: 'float16',
      timestep: 'float16',
      guidance: 'cfg',
    },
    reference: { fixedMs: 15500, perStepMs: 10460, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 25, hint: '20–30 is the sweet spot.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 6,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([1024, 768]),
        default: '1024x1024',
        hint: 'SDXL is made for 1024 × 1024; smaller is faster but less detailed.',
      },
      SEED,
    ],
  },
  {
    id: 'juggernaut-xl',
    name: 'Juggernaut XL',
    publisher: 'RunDiffusion',
    parameters: '3.5B',
    parameterCount: 3_500_000_000,
    rank: 8,
    summary:
      'The best-known photorealistic SDXL model: lifelike people, places and products at 1024 × 1024. About 25 steps with a negative prompt.',
    sizeLabel: '1024 × 1024',
    license: OPENRAIL_PP,
    precision: 'fp16',
    // Juggernaut XL v5, an Olive float16 export (float32 inputs and outputs); GroupNorm rewritten at load. The second text encoder's weights are inside its graph (the repo's model.onnx.data is unused).
    source: {
      host: 'huggingface',
      repo: 'softwareweaver/juggernaut-xl-v5-Olive-Onnx',
      revision: 'c264e131fa320a3066cbce9d371e2f4b6c3ed526',
    },
    pipeline: {
      kind: 'sdxl',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        vocab: {
          path: 'tokenizer_2/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer_2/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 246482723,
          sha256: 'b15ad54a02efef176f9555e5fd02b42a58ce2acad381394730a9957eb652f688',
        },
      },
      textEncoder2: {
        graph: {
          path: 'text_encoder_2/model.onnx',
          sizeBytes: 1390289158,
          sha256: '5aad83e2c6cd09b2c841b8f88aa6261dde7c92a24fdbe9b55421160c944717bd',
        },
      },
      unet: {
        graph: { path: 'unet/model.onnx', sizeBytes: 1251113, sha256: '4c3e98c1f76da4d1dc18adbf667fe4f28b69d6ed5c21f88e3b9d970fc0fc666d' },
        data: [
          {
            path: 'unet/model.onnx.data',
            sizeBytes: 5134903040,
            sha256: '04387c81b8cb1657248e4a29db5c38bfeb4a81e07b3804ec1e6ef0e8f2bb826e',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 102364507,
          sha256: '5aff31c90c286576161d1e1650c6b813f132c868aedcbe6e380b41ad9c975d71',
        },
      },
      previewDecoder: { graph: TAESDXL },
      scheduler: { kind: 'euler', spacing: 'leading' },
      io: 'float32',
      timestep: 'float32',
      guidance: 'cfg',
    },
    reference: { fixedMs: 15880, perStepMs: 13210, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 25, hint: '20–30 is the sweet spot.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 15,
        step: 0.5,
        default: 6,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([1024, 768]),
        default: '1024x1024',
        hint: 'SDXL is made for 1024 × 1024; smaller is faster but less detailed.',
      },
      SEED,
    ],
  },
  {
    id: 'animagine-xl-3.1',
    name: 'Animagine XL 3.1',
    publisher: 'Cagliostro Lab',
    parameters: '3.5B',
    parameterCount: 3_500_000_000,
    rank: 7,
    summary:
      'Anime-style illustration built on SDXL: knows many anime characters and art styles, prompted with Danbooru-style tags (e.g. "1girl, solo, smile, masterpiece, best quality"). About 28 steps.',
    sizeLabel: '1024 × 1024',
    license: OPENRAIL_PP,
    precision: 'fp16',
    // Our own export (tools/export/export_sdxl.py): float16 text encoders and UNet (float32 inputs and outputs), float32 VAE decoder.
    source: {
      host: 'huggingface',
      repo: `${OWN_EXPORTS}/animagine-xl-3.1-ONNX-WebGPU`,
      revision: '0e02fe59b6bc07d94aaf953b2f4a5496e8036f6e',
    },
    pipeline: {
      kind: 'sdxl',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        vocab: {
          path: 'tokenizer_2/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'tokenizer_2/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '!',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 232118385,
          sha256: 'da3214a5761043fd9ba880e9acdb697641ca1b55dd107048e0b65f1ee595b2cc',
        },
      },
      textEncoder2: {
        graph: {
          path: 'text_encoder_2/model.onnx',
          sizeBytes: 1389834142,
          sha256: 'e311e1725287ee5f84707e9184e1eae88af66c85595a967f06db9f6bc1df1ff0',
        },
      },
      unet: {
        graph: { path: 'unet/model.onnx', sizeBytes: 3809086, sha256: '40575fb16a8434a33de7314094632f2e233911bb5d7a70d15590478b9dce35de' },
        data: [
          {
            path: 'unet/model.onnx.data',
            sizeBytes: 5134903040,
            sha256: 'c70f9349a8764fe8a80cee4c885f258a2ead53f9440d411ef5f970fbf4a95f22',
          },
        ],
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 198078716,
          sha256: '4ff36b2a02436e7caa5581d59e2c629a08ae5943398899c8db0c0dc4d1245803',
        },
      },
      previewDecoder: { graph: TAESDXL },
      scheduler: { kind: 'euler', spacing: 'leading', ancestral: true },
      io: 'float32',
      timestep: 'float32',
      guidance: 'cfg',
    },
    reference: { fixedMs: 32800, perStepMs: 13210, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 28, hint: 'The model card recommends about 28.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 12,
        step: 0.5,
        default: 6,
        hint: '5–7 works best; higher follows the prompt more closely.',
      },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: [
          ...square([1024]),
          { value: '832x1216', label: '832 × 1216 (portrait)' },
          { value: '1216x832', label: '1216 × 832 (landscape)' },
          ...square([768]),
        ],
        default: '1024x1024',
        hint: 'Made for about a million pixels; smaller is faster but less detailed.',
      },
      SEED,
    ],
  },
  {
    id: 'sd-3.5-medium',
    name: 'Stable Diffusion 3.5 Medium',
    publisher: 'Stability AI',
    parameters: '2.5B',
    parameterCount: 2_470_000_000,
    rank: 8,
    arena: 760,
    summary:
      'Stability AI’s current mid-size model: good prompt following and text in images, at 1024 × 1024. About 30 steps with guidance.',
    sizeLabel: '1024 × 1024',
    license: STABILITY_COMMUNITY,
    precision: 'fp16',
    // Stability's float16 export for AMD GPUs (Olive): T5 and CLIP-G 4-bit; GroupNorm rewritten at load.
    source: {
      host: 'huggingface',
      repo: 'stabilityai/stable-diffusion-3.5-medium_amdgpu',
      revision: '7fda8db0a85dd26abac622143e948f05ffafc30e',
    },
    pipeline: {
      kind: 'sd3',
      tokenizer: {
        vocab: {
          path: 'stable-diffusion-3.5-medium_io16/tokenizer/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'stable-diffusion-3.5-medium_io16/tokenizer/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        vocab: {
          path: 'stable-diffusion-3.5-medium_io16/tokenizer_2/vocab.json',
          sizeBytes: 1059962,
          sha256: 'e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349',
        },
        merges: {
          path: 'stable-diffusion-3.5-medium_io16/tokenizer_2/merges.txt',
          sizeBytes: 524619,
          sha256: '9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a',
        },
        pad: '!',
      },
      tokenizer3: {
        path: 'stable-diffusion-3.5-medium_io16/tokenizer_3/tokenizer.json',
        sizeBytes: 2424035,
        sha256: '652ffdfc379606bad8edfb653f92dcf28e2e5dbf1cdfe50d685d29b2cad12dd6',
      },
      textEncoder: {
        graph: {
          path: 'stable-diffusion-3.5-medium_io16/text_encoder/model.onnx',
          sizeBytes: 247502190,
          sha256: 'b1b60d6f79df6c7083d8401f58a9b0ed1f1fef6149f6ccceda4f1d8ceccae2f9',
        },
      },
      textEncoder2: {
        graph: {
          path: 'stable-diffusion-3.5-medium_io16/text_encoder_2/model.onnx',
          sizeBytes: 613552,
          sha256: '066377400ed5687322ca95431b7390385aa4cbcd0b4c2c04b4e519f920dfdc31',
        },
        data: [
          {
            path: 'stable-diffusion-3.5-medium_io16/text_encoder_2/model.onnx.data',
            sizeBytes: 452999680,
            sha256: 'ee00202b1fd802f8f0d4f2dfa3e6a44dd977dc39e3e28e1d40473dbb87aa644e',
          },
        ],
      },
      textEncoder3: {
        // T5 from AMD's FLUX.1 export (the same google/t5-v1_1-xxl): this export's own squares activations in float16
        // without T5's overflow clamp, which comes out as zeros on WebGPU.
        graph: {
          path: 't5/model.onnx',
          sizeBytes: 717636,
          sha256: '6c817e5aa821d609cac4cafbf1ac8a093e1707cbd0b46cee3f35a258130c7d41',
          from: {
            repo: 'amd/FLUX.1-schnell_io16_amdgpu',
            revision: '081a0064fb6ab894542a53745a00b7c8f434d8a7',
            path: 'text_encoder_2/model.onnx',
          },
        },
        data: [
          {
            path: 't5/model.onnx.data',
            sizeBytes: 2666934272,
            sha256: '021936d947827d4609243dd516bda15328dc768e4a2af428025deaebb80d58d9',
            from: {
              repo: 'amd/FLUX.1-schnell_io16_amdgpu',
              revision: '081a0064fb6ab894542a53745a00b7c8f434d8a7',
              path: 'text_encoder_2/model.onnx.data',
            },
          },
        ],
      },
      transformer: {
        graph: {
          path: 'stable-diffusion-3.5-medium_io16/transformer/model.onnx',
          sizeBytes: 1892082,
          sha256: '174ff669bd11bc878a8d3c1df08006693d6b431038f2b8580a72ae84677c5c58',
        },
        data: [
          {
            path: 'stable-diffusion-3.5-medium_io16/transformer/model.onnx.data',
            sizeBytes: 4939312128,
            sha256: 'dc222c29ed279e780554e79eb17e6c7e95c20a4195c140d7222b547bc1f59c65',
          },
        ],
        // Float32 maths (float16 overflows above 512²), with 8-bit weights so it stays near its float16 size.
        quantizeWeights: 8,
        computeFloat32: true,
      },
      vaeDecoder: {
        graph: {
          path: 'stable-diffusion-3.5-medium_io16/vae_decoder/model.onnx',
          sizeBytes: 102474595,
          sha256: 'd3fe331f4a2e886d857606277ab4f8b19f7d2bb4fcc9337349a9a8f12b9e9a9c',
        },
      },
      guidance: 'cfg',
    },
    reference: { fixedMs: 25680, perStepMs: 21680, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 30, hint: '28–40 steps give the most detail.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 10,
        step: 0.5,
        default: 4.5,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([1024, 768, 512]),
        default: '768x768',
        hint: '1024 × 1024 gives the most detail but takes about twice as long as 768 × 768.',
      },
      SEED,
    ],
  },
  {
    id: 'sd-3.5-large',
    name: 'Stable Diffusion 3.5 Large',
    publisher: 'Stability AI',
    parameters: '8.1B',
    parameterCount: 8_100_000_000,
    rank: 10,
    arena: 840,
    summary:
      'Stability AI’s largest open model: the best image quality and prompt following of the Stable Diffusion family. A 20 GB download for big GPUs; minutes per image.',
    sizeLabel: '1024 × 1024',
    license: STABILITY_COMMUNITY,
    precision: 'fp16',
    // Stability's float16 export for AMD GPUs (Olive): T5 and CLIP-G 4-bit; GroupNorm rewritten at load.
    source: {
      host: 'huggingface',
      repo: 'stabilityai/stable-diffusion-3.5-large_amdgpu',
      revision: '23a6a0dcc29749b943e8abff2a71b8d9651570ab',
    },
    pipeline: {
      kind: 'sd3',
      tokenizer: {
        vocab: {
          path: 'stable-diffusion-3-5-large_io16/tokenizer/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'stable-diffusion-3-5-large_io16/tokenizer/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        vocab: {
          path: 'stable-diffusion-3-5-large_io16/tokenizer_2/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'stable-diffusion-3-5-large_io16/tokenizer_2/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '!',
      },
      tokenizer3: {
        path: 'stable-diffusion-3-5-large_io16/tokenizer_3/tokenizer.json',
        sizeBytes: 2424035,
        sha256: '652ffdfc379606bad8edfb653f92dcf28e2e5dbf1cdfe50d685d29b2cad12dd6',
      },
      textEncoder: {
        graph: {
          path: 'stable-diffusion-3-5-large_io16/text_encoder/model.onnx',
          sizeBytes: 247502190,
          sha256: 'b1b60d6f79df6c7083d8401f58a9b0ed1f1fef6149f6ccceda4f1d8ceccae2f9',
        },
      },
      textEncoder2: {
        graph: {
          path: 'stable-diffusion-3-5-large_io16/text_encoder_2/model.onnx',
          sizeBytes: 613552,
          sha256: '066377400ed5687322ca95431b7390385aa4cbcd0b4c2c04b4e519f920dfdc31',
        },
        data: [
          {
            path: 'stable-diffusion-3-5-large_io16/text_encoder_2/model.onnx.data',
            sizeBytes: 452999680,
            sha256: 'ee00202b1fd802f8f0d4f2dfa3e6a44dd977dc39e3e28e1d40473dbb87aa644e',
          },
        ],
      },
      textEncoder3: {
        // T5 from AMD's FLUX.1 export (the same google/t5-v1_1-xxl): this export's own squares activations in float16
        // without T5's overflow clamp, which comes out as zeros on WebGPU.
        graph: {
          path: 't5/model.onnx',
          sizeBytes: 717636,
          sha256: '6c817e5aa821d609cac4cafbf1ac8a093e1707cbd0b46cee3f35a258130c7d41',
          from: {
            repo: 'amd/FLUX.1-schnell_io16_amdgpu',
            revision: '081a0064fb6ab894542a53745a00b7c8f434d8a7',
            path: 'text_encoder_2/model.onnx',
          },
        },
        data: [
          {
            path: 't5/model.onnx.data',
            sizeBytes: 2666934272,
            sha256: '021936d947827d4609243dd516bda15328dc768e4a2af428025deaebb80d58d9',
            from: {
              repo: 'amd/FLUX.1-schnell_io16_amdgpu',
              revision: '081a0064fb6ab894542a53745a00b7c8f434d8a7',
              path: 'text_encoder_2/model.onnx.data',
            },
          },
        ],
      },
      transformer: {
        graph: {
          path: 'stable-diffusion-3-5-large_io16/unet/model.onnx',
          sizeBytes: 3633924,
          sha256: '908013b7751e627f5fdb2d7e0de42c8434c3edfe6798c03df2d5c0be109da894',
        },
        data: [
          {
            path: 'stable-diffusion-3-5-large_io16/unet/model.onnx.data',
            sizeBytes: 16292541952,
            sha256: '270c51dbbb56e9e6489a2879d8ec7929f45a40747c2a32cc6ab5a08fbc0a4c38',
          },
        ],
        // 16 GB of float16 weights: 8-bit at load, so it fits in about 9 GB; float32 maths (float16 overflows above 512²).
        quantizeWeights: 8,
        computeFloat32: true,
      },
      vaeDecoder: {
        graph: {
          path: 'stable-diffusion-3-5-large_io16/vae_decoder/model.onnx',
          sizeBytes: 102474595,
          sha256: 'd3fe331f4a2e886d857606277ab4f8b19f7d2bb4fcc9337349a9a8f12b9e9a9c',
        },
      },
      guidance: 'cfg',
    },
    reference: { fixedMs: 25610, perStepMs: 56660, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      NEGATIVE,
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 28, hint: 'About 28 steps.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 10,
        step: 0.5,
        default: 3.5,
        hint: 'How closely to follow the prompt (and avoid the negative prompt).',
      },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([1024, 768]),
        default: '768x768',
        hint: '1024 × 1024 gives the most detail but takes about twice as long as 768 × 768.',
      },
      SEED,
    ],
  },
  {
    id: 'sd-3.5-large-turbo',
    name: 'Stable Diffusion 3.5 Large Turbo',
    publisher: 'Stability AI',
    parameters: '8.1B',
    parameterCount: 8_100_000_000,
    rank: 9,
    arena: 829,
    summary:
      'Stable Diffusion 3.5 Large distilled to 4 steps without guidance: nearly its quality, several times faster. A 20 GB download for big GPUs.',
    sizeLabel: '1024 × 1024',
    license: STABILITY_COMMUNITY,
    precision: 'fp16',
    // Stability's float16 export for AMD GPUs (Olive): T5 and CLIP-G 4-bit; GroupNorm rewritten at load.
    source: {
      host: 'huggingface',
      repo: 'stabilityai/stable-diffusion-3.5-large-turbo_amdgpu',
      revision: 'c103af3299c110cec2dabfc44b21cac0fdee6788',
    },
    pipeline: {
      kind: 'sd3',
      tokenizer: {
        vocab: {
          path: 'stable-diffusion-3-5-large-turbo_io32/tokenizer/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'stable-diffusion-3-5-large-turbo_io32/tokenizer/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        vocab: {
          path: 'stable-diffusion-3-5-large-turbo_io32/tokenizer_2/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'stable-diffusion-3-5-large-turbo_io32/tokenizer_2/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '!',
      },
      tokenizer3: {
        path: 'stable-diffusion-3-5-large-turbo_io32/tokenizer_3/tokenizer.json',
        sizeBytes: 2424035,
        sha256: '652ffdfc379606bad8edfb653f92dcf28e2e5dbf1cdfe50d685d29b2cad12dd6',
      },
      textEncoder: {
        graph: {
          path: 'stable-diffusion-3-5-large-turbo_io32/text_encoder/model.onnx',
          sizeBytes: 247504552,
          sha256: '538f0edbf93f7a4179e84564d2163efef1a6f7d31c528c8577d7ae76b654458f',
        },
      },
      textEncoder2: {
        graph: {
          path: 'stable-diffusion-3-5-large-turbo_io32/text_encoder_2/model.onnx',
          sizeBytes: 619153,
          sha256: 'ce8888587921e7ee9a2ee0c60a9b0cf9f38bad77399acba8419aba8836ea084e',
        },
        data: [
          {
            path: 'stable-diffusion-3-5-large-turbo_io32/text_encoder_2/model.onnx.data',
            sizeBytes: 452999680,
            sha256: 'ee00202b1fd802f8f0d4f2dfa3e6a44dd977dc39e3e28e1d40473dbb87aa644e',
          },
        ],
      },
      textEncoder3: {
        // T5 from AMD's FLUX.1 export (the same google/t5-v1_1-xxl): this export's own squares activations in float16
        // without T5's overflow clamp, which comes out as zeros on WebGPU.
        graph: {
          path: 't5/model.onnx',
          sizeBytes: 717636,
          sha256: '6c817e5aa821d609cac4cafbf1ac8a093e1707cbd0b46cee3f35a258130c7d41',
          from: {
            repo: 'amd/FLUX.1-schnell_io16_amdgpu',
            revision: '081a0064fb6ab894542a53745a00b7c8f434d8a7',
            path: 'text_encoder_2/model.onnx',
          },
        },
        data: [
          {
            path: 't5/model.onnx.data',
            sizeBytes: 2666934272,
            sha256: '021936d947827d4609243dd516bda15328dc768e4a2af428025deaebb80d58d9',
            from: {
              repo: 'amd/FLUX.1-schnell_io16_amdgpu',
              revision: '081a0064fb6ab894542a53745a00b7c8f434d8a7',
              path: 'text_encoder_2/model.onnx.data',
            },
          },
        ],
      },
      transformer: {
        graph: {
          path: 'stable-diffusion-3-5-large-turbo_io32/unet/model.onnx',
          sizeBytes: 3634824,
          sha256: '03cebacb95c101dd68b2f4ee1d6e0d6662eb774be6c31145d57601e5f8094c95',
        },
        data: [
          {
            path: 'stable-diffusion-3-5-large-turbo_io32/unet/model.onnx.data',
            sizeBytes: 16292541952,
            sha256: '4d125315cc4dbeb2f3e1850a39b60b1122ff15f3174a4646c92d5a2ebe33591e',
          },
        ],
        // 16 GB of float16 weights: 8-bit at load, so it fits in about 9 GB; float32 maths (float16 overflows above 512²).
        quantizeWeights: 8,
        computeFloat32: true,
      },
      vaeDecoder: {
        graph: {
          path: 'stable-diffusion-3-5-large-turbo_io32/vae_decoder/model.onnx',
          sizeBytes: 102474706,
          sha256: '1305d9deb772f3d4a60a416cfac0d4e930534906c1aae73cbbe7210a63ca6350',
        },
      },
    },
    reference: { fixedMs: 20670, perStepMs: 27700, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      { kind: 'range', id: 'steps', label: 'Steps', min: 1, max: 8, step: 1, default: 4, hint: 'It’s made for 4 steps.' },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([1024, 768]),
        default: '768x768',
        hint: '1024 × 1024 gives the most detail but takes about twice as long as 768 × 768.',
      },
      SEED,
    ],
  },
  {
    id: 'flux2-klein-4b',
    name: 'FLUX.2 klein 4B',
    publisher: 'Black Forest Labs',
    parameters: '4B',
    parameterCount: 4_000_000_000,
    summary:
      'The small model of the FLUX.2 family, the best-known open image models: sharp, well-composed images in four steps. A 4.9 GB download.',
    sizeLabel: '512 × 512 to 1024 × 1024',
    license: { name: 'Apache 2.0', url: 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B' },
    rank: 9,
    precision: 'q4f16',
    // Our own export of the transformer (tools/export/export_flux2.py: 4-bit, float32 residual stream). The text encoder
    // and the decoder (in stages) are MarkShark2's WebGPU export, whose transformer fails on some GPUs.
    source: {
      host: 'huggingface',
      repo: `${OWN_EXPORTS}/FLUX.2-klein-4B-ONNX-WebGPU`,
      revision: '943297f6e9f8501ac8d6bfa7a43836cc8a93cee2',
    },
    pipeline: {
      kind: 'flux2',
      tokenizer: {
        path: 'markshark2/tokenizer/tokenizer.json',
        sizeBytes: 11422650,
        sha256: 'be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506',
        from: {
          repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
          revision: '929085adfad60491e9af05095cd280955fc0c29d',
          path: 'tokenizer/tokenizer.json',
        },
      },
      textEncoder: {
        optimization: 'disabled',
        freeDimensions: { batch: 1, text_seq: 512 },
        graph: {
          path: 'markshark2/flux2-klein-4b-text-encoder-q4.onnx',
          sizeBytes: 1844725,
          sha256: '8da1284e3834fa8fab918536bfd0fed39f3195f24110db159c7e60e0b5bde3c1',
          from: {
            repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
            revision: '929085adfad60491e9af05095cd280955fc0c29d',
            path: 'flux2-klein-4b-text-encoder-q4.onnx',
          },
        },
        data: [
          {
            path: 'markshark2/text-encoder-q4-00001.onnx_data',
            sizeBytes: 2136381440,
            sha256: '990bda47be7775658434c1cb775efdbcd4bc9eff1166c57e0ecee753f38f54db',
            from: {
              repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
              revision: '929085adfad60491e9af05095cd280955fc0c29d',
              path: 'text-encoder-q4-00001.onnx_data',
            },
          },
          {
            path: 'markshark2/text-encoder-q4-00002.onnx_data',
            sizeBytes: 430080000,
            sha256: '1dfacf3d2920e71764d4e115c45c670e4694e6d9997eeae94584634d0f30f88c',
            from: {
              repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
              revision: '929085adfad60491e9af05095cd280955fc0c29d',
              path: 'text-encoder-q4-00002.onnx_data',
            },
          },
        ],
      },
      transformer: {
        graph: { path: 'transformer.onnx', sizeBytes: 272881, sha256: '27e21af285287ef8ef20a6f01abc6bf229d7afa163212da5cb2a24c483cf9b3d' },
        data: [
          {
            path: 'transformer.onnx.data',
            sizeBytes: 2216276992,
            sha256: 'ff637423748a58d63f8101fdd045e31850067e33fead29556fe732b0b220cb12',
          },
        ],
      },
      vaePre: {
        graph: {
          path: 'markshark2/flux2-klein-4b-vae-decoder-pre-attn-fp16.onnx',
          sizeBytes: 11352251,
          sha256: '11010d757c8bbdc52def13b169db282f4c71ee65d2611423875c7c59fcbc4ec7',
          from: {
            repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
            revision: '929085adfad60491e9af05095cd280955fc0c29d',
            path: 'flux2-klein-4b-vae-decoder-pre-attn-fp16.onnx',
          },
        },
      },
      vaeAttention: {
        graph: {
          path: 'markshark2/flux2-klein-4b-vae-decoder-attn-chunk-fp16.onnx',
          sizeBytes: 480,
          sha256: '1148726ed473c70481d89d2d68af082c36cf51def1e60b1cb6d465ffff364bd0',
          from: {
            repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
            revision: '929085adfad60491e9af05095cd280955fc0c29d',
            path: 'flux2-klein-4b-vae-decoder-attn-chunk-fp16.onnx',
          },
        },
      },
      vaePost: [
        {
          graph: {
            path: 'markshark2/flux2-klein-4b-vae-decoder-post-stage0-fp16.onnx',
            sizeBytes: 85202062,
            sha256: 'f4b7658c0a49fa240a4ff672a3a5528167fc21e33ff3f7416c1d13910463151a',
            from: {
              repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
              revision: '929085adfad60491e9af05095cd280955fc0c29d',
              path: 'flux2-klein-4b-vae-decoder-post-stage0-fp16.onnx',
            },
          },
        },
        {
          graph: {
            path: 'markshark2/flux2-klein-4b-vae-decoder-post-stage1-fp16.onnx',
            sizeBytes: 2141312,
            sha256: 'e13bf34b75d7586012e388794fd129b2ee7619293510ca36ffde68b1edf00ca1',
            from: {
              repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
              revision: '929085adfad60491e9af05095cd280955fc0c29d',
              path: 'flux2-klein-4b-vae-decoder-post-stage1-fp16.onnx',
            },
          },
        },
        {
          graph: {
            path: 'markshark2/flux2-klein-4b-vae-decoder-post-stage2-fp16.onnx',
            sizeBytes: 1196419,
            sha256: '0f68a7af176f701170adab69dcb38ba8a6825dc2d006bf7d3eead90d38d82736',
            from: {
              repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
              revision: '929085adfad60491e9af05095cd280955fc0c29d',
              path: 'flux2-klein-4b-vae-decoder-post-stage2-fp16.onnx',
            },
          },
        },
        {
          graph: {
            path: 'markshark2/flux2-klein-4b-vae-decoder-post-stage3-fp16.onnx',
            sizeBytes: 10460,
            sha256: 'a8a655923a579caf2afb465a174184bacd17794f29e7ce35dd15d8d565e73461',
            from: {
              repo: 'MarkShark2/flux2-klein-4b-onnx-webgpu-q4',
              revision: '929085adfad60491e9af05095cd280955fc0c29d',
              path: 'flux2-klein-4b-vae-decoder-post-stage3-fp16.onnx',
            },
          },
        },
      ],
    },
    reference: { fixedMs: 12300, perStepMs: 6550, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([512, 768, 1024]),
        default: '512x512',
        hint: 'Larger sizes take longer.',
      },
      { kind: 'range', id: 'steps', label: 'Steps', min: 2, max: 8, step: 1, default: 4, hint: 'It’s tuned for 4.' },
      SEED,
    ],
  },
  {
    id: 'flux.1-schnell',
    name: 'FLUX.1 schnell',
    publisher: 'Black Forest Labs',
    parameters: '12B',
    parameterCount: 11_900_000_000,
    rank: 10,
    arena: 803,
    summary:
      'Black Forest Labs’ fast FLUX: excellent images and prompt following in 4 steps, with readable text. A 27 GB download for big GPUs.',
    sizeLabel: '1024 × 1024',
    license: { name: 'Apache 2.0', url: 'https://huggingface.co/black-forest-labs/FLUX.1-schnell' },
    precision: 'fp16',
    // AMD's float16 GPU export: T5 4-bit; the 24 GB transformer quantized to 4 bits at load (about 6.7 GB; at 8 bits, 12.6 GB
    // with T5 and the activations ran out of memory on a typical laptop).
    source: { host: 'huggingface', repo: 'amd/FLUX.1-schnell_io16_amdgpu', revision: '081a0064fb6ab894542a53745a00b7c8f434d8a7' },
    pipeline: {
      kind: 'flux1',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        path: 'tokenizer_2/tokenizer.json',
        sizeBytes: 2424235,
        sha256: 'f5dfec163765e18e270537fe896c49f5fad74db1525641d9b255a3008b999596',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 313594,
          sha256: '4fa553e25a5b293966fd831fbd69f3a391acd59be04b85c416330809e4b07233',
        },
        data: [
          {
            path: 'text_encoder/model.onnx.data',
            sizeBytes: 246120960,
            sha256: 'd2015263f98ae9f6e0ea233756bd5ea51ddb1f36ec6a2dd862e59d82611d1db7',
          },
        ],
      },
      textEncoder2: {
        graph: {
          path: 'text_encoder_2/model.onnx',
          sizeBytes: 717636,
          sha256: '6c817e5aa821d609cac4cafbf1ac8a093e1707cbd0b46cee3f35a258130c7d41',
        },
        data: [
          {
            path: 'text_encoder_2/model.onnx.data',
            sizeBytes: 2666934272,
            sha256: '021936d947827d4609243dd516bda15328dc768e4a2af428025deaebb80d58d9',
          },
        ],
      },
      transformer: {
        graph: {
          path: 'transformer/model.onnx',
          sizeBytes: 3681116,
          sha256: '594ca1e01195e2e3e5c5c5f8616c9b3b0488cded04b0390b8ae3f7f05a1b762d',
        },
        data: [
          {
            path: 'transformer/model.onnx.data',
            sizeBytes: 23782318080,
            sha256: '65418c2ffc1139980d173c5136de7c82895cde1fa53234aa19cd8fc90b507fcc',
          },
        ],
        quantizeWeights: 4,
        // Float32 maths: its activations overflow float16 (black images).
        computeFloat32: true,
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 134332,
          sha256: '5d9921d5ceab77f8ce59a7b077c71dba9652722e36610c0ccd4aa817796e26b5',
        },
        data: [
          {
            path: 'vae_decoder/model.onnx.data',
            sizeBytes: 198172160,
            sha256: 'b7851af2be3f720d2e383974f0421a9a0ea7d5c6863cb23c1f001d5847d31c40',
          },
        ],
      },
      textTokens: 256,
    },
    reference: { fixedMs: 19920, perStepMs: 55230, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      { kind: 'range', id: 'steps', label: 'Steps', min: 1, max: 8, step: 1, default: 4, hint: 'It’s made for 1–4 steps.' },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([1024, 768, 512]),
        default: '768x768',
        hint: '1024 × 1024 gives the most detail but takes about twice as long as 768 × 768.',
      },
      SEED,
    ],
  },
  {
    id: 'flux.1-dev',
    name: 'FLUX.1 dev',
    publisher: 'Black Forest Labs',
    parameters: '12B',
    parameterCount: 11_900_000_000,
    rank: 11,
    arena: 841,
    summary:
      'The most popular open image model: FLUX at full quality, about 28 steps; for non-commercial use only. A 27 GB download for big GPUs; minutes per image.',
    sizeLabel: '1024 × 1024',
    license: {
      name: 'FLUX.1 [dev] Non-Commercial License',
      url: 'https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md',
    },
    precision: 'fp16',
    // AMD's float16 GPU export: T5 4-bit; the 24 GB transformer quantized to 4 bits at load (about 6.7 GB; at 8 bits, 12.6 GB
    // with T5 and the activations ran out of memory on a typical laptop).
    source: { host: 'huggingface', repo: 'amd/FLUX.1-dev_io16_amdgpu', revision: 'ee67d5d8c4122fac5c4ebceade98171126b733e0' },
    pipeline: {
      kind: 'flux1',
      tokenizer: {
        vocab: {
          path: 'tokenizer/vocab.json',
          sizeBytes: 1109372,
          sha256: 'fd67774a869730a6b27bf53d3e434e72054f2a825873c7af3bece183bb1f791e',
        },
        merges: {
          path: 'tokenizer/merges.txt',
          sizeBytes: 573514,
          sha256: '0bc7695944744789d6fd6d0ab754bcbbb9e36f1c182df0a006d619ce70a1e052',
        },
        pad: '<|endoftext|>',
      },
      tokenizer2: {
        path: 'tokenizer_2/tokenizer.json',
        sizeBytes: 2424235,
        sha256: 'f5dfec163765e18e270537fe896c49f5fad74db1525641d9b255a3008b999596',
      },
      textEncoder: {
        graph: {
          path: 'text_encoder/model.onnx',
          sizeBytes: 313594,
          sha256: '4fa553e25a5b293966fd831fbd69f3a391acd59be04b85c416330809e4b07233',
        },
        data: [
          {
            path: 'text_encoder/model.onnx.data',
            sizeBytes: 246120960,
            sha256: 'd2015263f98ae9f6e0ea233756bd5ea51ddb1f36ec6a2dd862e59d82611d1db7',
          },
        ],
      },
      textEncoder2: {
        graph: {
          path: 'text_encoder_2/model.onnx',
          sizeBytes: 528124,
          sha256: '7311de30896c46419394de3e0f1f0166d36a0e1cd0b2fde4ae1e1536653dde6b',
        },
        data: [
          {
            path: 'text_encoder_2/model.onnx.data',
            sizeBytes: 2666934272,
            sha256: '021936d947827d4609243dd516bda15328dc768e4a2af428025deaebb80d58d9',
          },
        ],
      },
      transformer: {
        graph: {
          path: 'transformer/model.onnx',
          sizeBytes: 3688299,
          sha256: '7fb547c2369c165652ec2ea4887e2214930a243a79f13c5f3a50d67f4c91d4b9',
        },
        data: [
          {
            path: 'transformer/model.onnx.data',
            sizeBytes: 23802777600,
            sha256: '0ce07d65cd98af396a880f2c53fe79c54cff717ccef15fbf06700a7a50dea803',
          },
        ],
        quantizeWeights: 4,
        // Float32 maths: its activations overflow float16 (black images).
        computeFloat32: true,
      },
      vaeDecoder: {
        graph: {
          path: 'vae_decoder/model.onnx',
          sizeBytes: 134332,
          sha256: '5d9921d5ceab77f8ce59a7b077c71dba9652722e36610c0ccd4aa817796e26b5',
        },
        data: [
          {
            path: 'vae_decoder/model.onnx.data',
            sizeBytes: 198172160,
            sha256: 'b7851af2be3f720d2e383974f0421a9a0ea7d5c6863cb23c1f001d5847d31c40',
          },
        ],
      },
      textTokens: 512,
      shiftSchedule: true,
      guidanceInput: true,
    },
    reference: { fixedMs: 34560, perStepMs: 76820, computeGflops: 809 },
    requiredGpuFeatures: ['shader-f16'],
    inputs: [
      { kind: 'range', id: 'steps', label: 'Steps', min: 10, max: 50, step: 1, default: 28, hint: '28–50 steps.' },
      {
        kind: 'range',
        id: 'guidance',
        label: 'Guidance',
        min: 1,
        max: 10,
        step: 0.5,
        default: 3.5,
        hint: 'How closely to follow the prompt.',
      },
      {
        kind: 'select',
        id: 'size',
        label: 'Size',
        options: square([1024, 768, 512]),
        default: '768x768',
        hint: '1024 × 1024 gives the most detail but takes about twice as long as 768 × 768.',
      },
      SEED,
    ],
  },
];

export function findImageModel(id: string): ImageModelSpec | undefined {
  return IMAGE_MODELS.find((m) => m.id === id);
}

/** A model that runs in the app (not a preview). */
export function runsInApp(model: ImageModelSpec): boolean {
  return model.pipeline !== undefined && model.source !== undefined;
}

/** Every file of a model that runs: tokenizers and small graphs first, the largest last. */
export function imageModelFiles(model: ImageModelSpec): ArtifactFile[] {
  const files: ArtifactFile[] = [];
  const walk = (value: unknown) => {
    if (!value || typeof value !== 'object') return;
    if ('sha256' in value && 'path' in value) files.push(value as ArtifactFile);
    else for (const v of Object.values(value)) walk(v);
  };
  walk(model.pipeline);
  return [...new Set(files)].sort((a, b) => a.sizeBytes - b.sizeBytes);
}

/** Every graph of a model's pipeline, by role (a list of graphs as role0, role1, …). */
export function imageGraphs(pipeline: ImagePipeline): Record<string, ImageGraph> {
  const isGraph = (v: unknown): v is ImageGraph => typeof v === 'object' && v !== null && 'graph' in v;
  const out: Record<string, ImageGraph> = {};
  for (const [role, value] of Object.entries(pipeline)) {
    if (isGraph(value)) out[role] = value;
    else if (Array.isArray(value)) value.forEach((g, i) => isGraph(g) && (out[`${role}${i}`] = g));
  }
  return out;
}

export function imageModelBytes(model: ImageModelSpec): number {
  return imageModelFiles(model).reduce((n, f) => n + f.sizeBytes, 0);
}

export function defaultSettings(model: ImageModelSpec): ImageSettings {
  const settings: ImageSettings = {};
  for (const input of model.inputs) {
    if (input.kind === 'text') settings[input.id] = '';
    else if (input.kind === 'seed') settings[input.id] = -1;
    else settings[input.id] = input.default;
  }
  return settings;
}

/** Output width and height for these settings. */
export function outputSize(model: ImageModelSpec, settings: ImageSettings): { width: number; height: number } {
  const chosen = typeof settings.size === 'string' ? settings.size : null;
  const [w, h] = (chosen ?? model.sizeLabel.split(' or ')[0]!.replace(/ × /, 'x')).split('x').map(Number);
  return { width: w || 512, height: h || 512 };
}
