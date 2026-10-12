// Image generation models. Each describes the inputs it takes beyond the prompt, so the workspace builds its
// settings from the model instead of assuming one set: a distilled model like SD Turbo takes a seed and a step or
// two, Stable Diffusion 1.5 also takes a negative prompt, a size and guidance. Models with `files` run in the app
// (src/runtime/image); the others are previews (placeholder images) until they do.

import type { ArtifactFile, Precision } from './models';

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

export interface ImageModelSpec {
  id: string;
  name: string;
  publisher: string;
  parameters: string;
  /** Total parameters, for ordering by size. */
  parameterCount: number;
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
   * Times on a reference device (Apple M4, 24 GB, Chrome) and its device-test compute, for estimates elsewhere:
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
  {
    id: 'sd-turbo',
    name: 'SD Turbo',
    publisher: 'Stability AI',
    parameters: '1B',
    parameterCount: 1_000_000_000,
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
