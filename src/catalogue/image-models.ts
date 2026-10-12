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
  spacing: 'trailing' | 'linspace';
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

/** Every graph of a model's pipeline, by role. */
export function imageGraphs(pipeline: ImagePipeline): Record<string, ImageGraph> {
  return Object.fromEntries(
    Object.entries(pipeline).filter((e): e is [string, ImageGraph] => typeof e[1] === 'object' && e[1] !== null && 'graph' in e[1]),
  );
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
