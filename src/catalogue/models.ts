// Model catalogue: stable model identities, separate from the artifacts that
// execute them. A model can have several artifacts (engines and precisions),
// listed in order of preference; the runtime picks the first this device can
// run (see src/runtime/compatibility.ts).
//
// Preference: a WebLLM build where one exists (lighter and faster, see
// info/devlog.md), then this app's own ONNX engine, which is also the fallback
// on devices where WebLLM fails.
//
// Generated pins: WebLLM builds by tools/webllm/make_catalogue.py
// (webllm-builds.ts); ONNX artifact entries are printed by
// tools/reference/onnx_artifact.py from the files tools/reference/fetch_model.py
// pins, and each is checked against its reference (tools/reference/make_fixtures.py).

import type { PromptFormatKind } from '../tokenizer/prompt-format';
import { WEBLLM_BUILDS } from './webllm-builds';

/** 4-bit weights with fp16 maths ('q4f16') or fp32 maths ('q4', for GPUs without fp16 shaders); 'q3f16': 3-bit (70B). */
export type Precision = 'q4f16' | 'q4' | 'q3f16';

/**
 * Which runtime runs an artifact: this app's own ONNX Runtime adapter
 * (src/runtime/onnx), or WebLLM's compiled MLC models (src/runtime/webllm).
 */
export type EngineId = 'onnx-webgpu' | 'webllm';

export interface ArtifactFile {
  /** Path within the pinned repository revision. */
  path: string;
  sizeBytes: number;
  sha256: string;
}

interface ArtifactBase {
  id: string;
  precision: Precision;
  /** WebGPU features the artifact needs (e.g. fp16 shaders). */
  requiredGpuFeatures: GPUFeatureName[];
}

export interface OnnxArtifact extends ArtifactBase {
  engine: 'onnx-webgpu';
  format: 'onnx';
  graph: ArtifactFile;
  /**
   * Weight files the graph refers to by file name. Exports over 2 GB keep their
   * weights outside the graph (ONNX's protobuf limit), split into ~2 GB parts.
   */
  externalData?: ArtifactFile[];
  /**
   * A separate token-embedding graph whose output (`inputs_embeds`) the decoder
   * takes instead of token ids (Qwen3.5's exports).
   */
  embeddings?: { graph: ArtifactFile; externalData: ArtifactFile[] };
  /**
   * Element type of the KV-cache tensors, used for memory estimates. The
   * runtime reads the real layout from the loaded graph and checks it.
   */
  kvCacheType: 'float16' | 'float32';
  /**
   * Bytes of the graph's largest single weight tensor (the word-embedding table
   * in our models). A GPU whose maximum buffer size is smaller can't run it.
   */
  largestTensorBytes: number;
  /**
   * Compute operations in the graph (excluding shape bookkeeping). The runtime
   * runs roughly this many GPU jobs per token, which sets the fixed per-token
   * cost; unfused exports have far more than onnxruntime-genai ones.
   */
  computeOps: number;
}

/** A file of a WebLLM build, within its pinned repository revision. */
export interface WebLlmFile {
  path: string;
  sizeBytes: number;
  /** For weight shards: the MD5 listed in the build's tensor-cache.json. */
  md5?: string;
  /** For the tokenizer: its SHA-256. */
  sha256?: string;
}

/** A pinned WebLLM (MLC) build; generated into webllm-builds.ts by tools/webllm/make_catalogue.py. */
export interface WebLlmBuild {
  repo: string;
  revision: string;
  /** The compiled model library (WebGPU kernels), pinned to a commit, with its SRI hash. */
  modelLib: { url: string; sizeBytes: number; integrity: string };
  /** SRI hashes WebLLM checks itself: config and tokenizer. */
  integrity: { config: string; tokenizer: Record<string, string> };
  /** The tokenizer file WebLLM downloads (tokenizer.json where there is one). */
  tokenizer: WebLlmFile;
  /** Every file WebLLM downloads, plus the model library. */
  downloadBytes: number;
  /** The weight shards (what each generated token reads). */
  weightBytes: number;
  /** GPU memory WebLLM states the build needs at its default context. */
  vramMB: number;
  /** The build's maximum context; the app loads it with `generation.contextTokens`. */
  contextWindow: number;
  prefillChunk: number | null;
  vocabSize: number;
  convTemplate: string;
  // Every file of a build, with shard MD5s, is in webllm-files.ts (worker only).
}

export interface WebLlmArtifact extends ArtifactBase {
  engine: 'webllm';
  format: 'mlc';
  /** WebLLM's model id, e.g. "Qwen3-0.6B-q4f16_1-MLC". */
  modelId: string;
  build: WebLlmBuild;
}

export type ModelArtifact = OnnxArtifact | WebLlmArtifact;

export interface ModelArchitecture {
  family: 'llama' | 'qwen2' | 'qwen3' | 'qwen3.5' | 'gemma2' | 'phi3';
  /** Total parameters, including embeddings (used for compute estimates). */
  parameterCount: number;
  layers: number;
  hiddenSize: number;
  kvHeads: number;
  headDim: number;
  vocabSize: number;
  maxContextTokens: number;
}

/** A score published by the model's makers (not measured by this app). */
export interface PublishedScore {
  benchmark: string;
  score: number;
  source: string;
}

export interface ModelQuality {
  /**
   * Relative answer quality among catalogue models (higher is better), judged
   * from the published scores below. Used to order recommendations.
   */
  rank: number;
  summary: string;
  scores: PublishedScore[];
}

export interface GenerationDefaults {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
  repetitionWindow: number;
  /** Context the app uses by default; may be smaller than the model's maximum. */
  contextTokens: number;
}

export interface ModelSpec {
  id: string;
  name: string;
  publisher: string;
  parameters: string;
  license: string;
  task: 'chat';
  /**
   * Repository of the model's ONNX files, which the ONNX store keys saved files
   * by. For a model with only WebLLM builds: its main WebLLM build.
   */
  source: { host: 'huggingface'; repo: string; revision: string };
  architecture: ModelArchitecture;
  /** tokenizer.json in `source`. */
  tokenizer: ArtifactFile;
  promptFormat: {
    /**
     * The format this app builds for the ONNX engine (see
     * tokenizer/prompt-format.ts); 'model-template': only
     * through the engine's own template (WebLLM-only models).
     */
    kind: PromptFormatKind | 'model-template';
    defaultSystemPrompt?: string;
    stopTokens: string[];
    /** Qwen3: answer without a reasoning phase (the template's enable_thinking=false). */
    disableThinking?: boolean;
    /** The template trims each message (Qwen3.5). */
    trimContent?: boolean;
  };
  generation: GenerationDefaults;
  quality: ModelQuality;
  /**
   * Listed without having run anywhere we could test it (too big for our machines);
   * shown to people as the reason. Never recommended, and left out of the device
   * test's memory check.
   */
  experimental?: string;
  /** Preferred first. */
  artifacts: ModelArtifact[];
}

const SMOLLM2_CARD = 'SmolLM2 model card (Hugging Face)';
const QWEN25_BLOG = 'Qwen2.5 announcement (Qwen team blog)';
const QWEN3_REPORT = 'Qwen3 technical report, table 20 (thinking off)';
const LLAMA32_CARD = 'Llama 3.2 model card (Meta)';
const QWEN35_CARD = 'Qwen3.5 model card (Hugging Face)';
const QWEN25_CODER_REPORT = 'Qwen2.5-Coder technical report, table 16';
const QWEN3_REPORT_4B = 'Qwen3 technical report, table 18 (thinking off)';
const LLAMA31_CARD = 'Llama 3.1 model card (Meta)';
const GEMMA2_REPORT = 'Gemma 2 technical report, tables 13, 14 and 17';
const PHI35_CARD = 'Phi-3.5-mini-instruct model card (Microsoft)';
const PHI4_CARD = 'Phi-4-mini-instruct model card (Microsoft)';
const CHATML_STOPS = ['<|im_end|>', '<|endoftext|>'];

export const SMOLLM2_360M_INSTRUCT: ModelSpec = {
  id: 'smollm2-360m-instruct',
  name: 'SmolLM2 360M',
  publisher: 'Hugging Face',
  parameters: '360M',
  license: 'Apache-2.0',
  task: 'chat',
  source: {
    host: 'huggingface',
    repo: 'HuggingFaceTB/SmolLM2-360M-Instruct',
    revision: 'a10cc1512eabd3dde888204e902eca88bddb4951',
  },
  architecture: {
    family: 'llama',
    parameterCount: 362_000_000,
    layers: 32,
    hiddenSize: 960,
    kvHeads: 5,
    headDim: 64,
    vocabSize: 49152,
    maxContextTokens: 8192,
  },
  tokenizer: {
    path: 'tokenizer.json',
    sizeBytes: 2104556,
    sha256: '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c',
  },
  promptFormat: {
    kind: 'chatml',
    defaultSystemPrompt: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
    stopTokens: ['<|im_end|>'],
  },
  // Temperature/top-p follow the model card. Without a repetition penalty, 7 of 18
  // test replies (6 prompts x 3 seeds) looped until the length limit; 1.1 over the
  // last 64 tokens removed all loops in that sample (see info/gameplan.md, 1.6).
  // Context is kept below the 8K maximum to bound KV-cache memory.
  generation: {
    maxNewTokens: 1024,
    temperature: 0.2,
    topP: 0.9,
    repetitionPenalty: 1.1,
    repetitionWindow: 64,
    contextTokens: 4096,
  },
  quality: {
    rank: 1,
    summary: 'The smallest and fastest. Fine for short, simple requests; weak at maths and longer reasoning.',
    scores: [
      { benchmark: 'IFEval (prompt/instruction average)', score: 41.0, source: SMOLLM2_CARD },
      { benchmark: 'MMLU (cloze)', score: 32.8, source: SMOLLM2_CARD },
      { benchmark: 'GSM8K (5-shot)', score: 7.43, source: SMOLLM2_CARD },
    ],
  },
  artifacts: [
    webllm('SmolLM2-360M-Instruct-q4f16_1-MLC'),
    webllm('SmolLM2-360M-Instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: {
        path: 'onnx/model_q4f16.onnx',
        sizeBytes: 272737275,
        sha256: 'cc63370efc2aca6d5307518b85162777132cc5b8d68eeb8154ea9b5fce09ad46',
      },
      kvCacheType: 'float16',
      largestTensorBytes: 94371840,
      computeOps: 485,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: {
        path: 'onnx/model_q4.onnx',
        sizeBytes: 387943246,
        sha256: 'a4ffda96e65beafc6f6cef0cbcf9fdc1cbdd79c230906bf3897d190547c7a596',
      },
      kvCacheType: 'float32',
      largestTensorBytes: 188743680, // assumed fp32 embedding (file not inspected)
      computeOps: 485, // assumed same graph as q4f16
    },
  ],
};

// Qwen models: sampling follows each model card's recommendation for chat (Qwen3
// with thinking off). Repetition penalty as in Qwen2.5's generation_config; a
// lighter one for Qwen3, whose card warns strong penalties cause language mixing.

export const QWEN25_05B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-0.5b-instruct',
  name: 'Qwen2.5 0.5B',
  publisher: 'Alibaba Qwen',
  parameters: '0.5B',
  license: 'Apache-2.0',
  task: 'chat',
  source: {
    host: 'huggingface',
    // The fused-attention export. The older onnx-community/Qwen2.5-0.5B-Instruct
    // export computes attention with plain MatMuls, which ONNX Runtime's WebGPU
    // kernel gets wrong on Adreno GPUs for some shapes (found on a Galaxy S24 Ultra:
    // identical inputs, wrong scores from 37 tokens on), and its q4f16 file
    // overflows fp16. This one runs 535 GPU jobs per token instead of 1,194.
    repo: 'onnx-community/Qwen2.5-0.5B-Instruct-ONNX-MHA',
    revision: 'b1e71b59c0c33ee49eb562303578961795624293',
  },
  architecture: {
    family: 'qwen2',
    parameterCount: 494_000_000,
    layers: 24,
    hiddenSize: 896,
    kvHeads: 2,
    headDim: 64,
    vocabSize: 151936,
    maxContextTokens: 32768,
  },
  tokenizer: {
    path: 'tokenizer.json',
    sizeBytes: 9116490,
    sha256: 'bb5953180a154511cf4d0fb1fc68a3433f1458aa6db3851ac2609cf41e7d9b5b',
  },
  promptFormat: {
    kind: 'chatml',
    defaultSystemPrompt: 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
    stopTokens: CHATML_STOPS,
  },
  generation: {
    maxNewTokens: 1024,
    temperature: 0.7,
    topP: 0.8,
    repetitionPenalty: 1.1,
    repetitionWindow: 64,
    contextTokens: 4096,
  },
  quality: {
    rank: 1,
    summary: 'Small and quick. Better at maths and code than SmolLM2, weaker at following instructions.',
    scores: [
      { benchmark: 'IFEval (strict prompt)', score: 27.9, source: QWEN25_BLOG },
      { benchmark: 'MMLU-Redux', score: 24.1, source: QWEN25_BLOG },
      { benchmark: 'GSM8K', score: 49.6, source: QWEN25_BLOG },
    ],
  },
  artifacts: [
    webllm('Qwen2.5-0.5B-Instruct-q4f16_1-MLC'),
    webllm('Qwen2.5-0.5B-Instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: {
        path: 'onnx/model_q4f16.onnx',
        sizeBytes: 478355493,
        sha256: '2a8e924345337b83796bd98bc130ab0ec0163188d50be55a1bf14b95808c55b8',
      },
      kvCacheType: 'float16',
      largestTensorBytes: 272269312,
      computeOps: 535,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: {
        path: 'onnx/model_q4.onnx',
        sizeBytes: 777326354,
        sha256: '60e105dee084373137dd810d0d5f47c3a0cfe14c2dd3c25f723fba3169e011f6',
      },
      kvCacheType: 'float32',
      largestTensorBytes: 544538624,
      computeOps: 535,
    },
  ],
};

export const QWEN3_06B: ModelSpec = {
  id: 'qwen3-0.6b',
  name: 'Qwen3 0.6B',
  publisher: 'Alibaba Qwen',
  parameters: '0.6B',
  license: 'Apache-2.0',
  task: 'chat',
  source: {
    host: 'huggingface',
    repo: 'onnx-community/Qwen3-0.6B-ONNX',
    revision: '1e0a4a196ecabdf9a879664110574563d3f372d3',
  },
  architecture: {
    family: 'qwen3',
    parameterCount: 596_000_000,
    layers: 28,
    hiddenSize: 1024,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 40960,
  },
  tokenizer: {
    path: 'tokenizer.json',
    sizeBytes: 9117040,
    sha256: 'e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5',
  },
  promptFormat: { kind: 'chatml', stopTokens: CHATML_STOPS, disableThinking: true },
  generation: {
    maxNewTokens: 1024,
    temperature: 0.7,
    topP: 0.8,
    repetitionPenalty: 1.05,
    repetitionWindow: 64,
    contextTokens: 4096,
  },
  quality: {
    rank: 2,
    summary: 'Newer and noticeably better at following instructions and general knowledge than the smaller models.',
    scores: [
      { benchmark: 'IFEval (strict prompt)', score: 54.5, source: QWEN3_REPORT },
      { benchmark: 'MMLU-Redux', score: 44.6, source: QWEN3_REPORT },
      { benchmark: 'MATH-500', score: 55.2, source: QWEN3_REPORT },
    ],
  },
  artifacts: [
    webllm('Qwen3-0.6B-q4f16_1-MLC'),
    webllm('Qwen3-0.6B-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: {
        path: 'onnx/model_q4f16.onnx',
        sizeBytes: 569789750,
        sha256: '9e33a5911974174761d0dfdcc0bec975d9c45af0eae5e9eb647b8ba9442a8f91',
      },
      kvCacheType: 'float16',
      largestTensorBytes: 311164928,
      computeOps: 481,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: {
        path: 'onnx/model_q4.onnx',
        sizeBytes: 919096585,
        sha256: 'd43d836fc5e240df9013733ccd214972c5d21bd9ec47e574e4f1e359cf90aed0',
      },
      kvCacheType: 'float32',
      largestTensorBytes: 622329856, // assumed fp32 embedding (file not inspected)
      computeOps: 481, // assumed same graph as q4f16
    },
  ],
};

export const SMOLLM2_17B_INSTRUCT: ModelSpec = {
  id: 'smollm2-1.7b-instruct',
  name: 'SmolLM2 1.7B',
  publisher: 'Hugging Face',
  parameters: '1.7B',
  license: 'Apache-2.0',
  task: 'chat',
  source: {
    host: 'huggingface',
    repo: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
    revision: '31b70e2e869a7173562077fd711b654946d38674',
  },
  // No grouped attention (32 KV heads): its conversation memory is ~0.8 GB at 4K context.
  architecture: {
    family: 'llama',
    parameterCount: 1_711_000_000,
    layers: 24,
    hiddenSize: 2048,
    kvHeads: 32,
    headDim: 64,
    vocabSize: 49152,
    maxContextTokens: 8192,
  },
  tokenizer: SMOLLM2_360M_INSTRUCT.tokenizer,
  promptFormat: SMOLLM2_360M_INSTRUCT.promptFormat,
  generation: SMOLLM2_360M_INSTRUCT.generation,
  quality: {
    rank: 3,
    summary: 'Clearly better at following instructions, knowledge and maths than the small models. Slower, and about a 1 GB download.',
    scores: [
      { benchmark: 'IFEval (prompt/instruction average)', score: 56.7, source: SMOLLM2_CARD },
      { benchmark: 'MT-Bench', score: 6.13, source: SMOLLM2_CARD },
      { benchmark: 'GSM8K (5-shot)', score: 48.2, source: SMOLLM2_CARD },
    ],
  },
  artifacts: [
    webllm('SmolLM2-1.7B-Instruct-q4f16_1-MLC'),
    webllm('SmolLM2-1.7B-Instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: {
        path: 'onnx/model_q4f16.onnx',
        sizeBytes: 1108730338,
        sha256: 'd94946187fb5f27579f3db4ba21fb7f7466c7cbd18956bd420d3981f75282f9c',
      },
      kvCacheType: 'float16',
      largestTensorBytes: 201326592,
      computeOps: 365,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: {
        path: 'onnx/model_q4.onnx',
        sizeBytes: 1411969607,
        sha256: '467b7b8f62d99f184d3628d24b8d65c151e331695f6e9ea997616c4e279e9a51',
      },
      kvCacheType: 'float32',
      largestTensorBytes: 402653184, // assumed fp32 embedding (file not inspected)
      computeOps: 365, // assumed same graph as q4f16
    },
  ],
};

const LLAMA3_FORMAT: ModelSpec['promptFormat'] = { kind: 'llama3', stopTokens: ['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'] };
const PHI_FORMAT: ModelSpec['promptFormat'] = { kind: 'phi', stopTokens: ['<|end|>', '<|endoftext|>'] };
// Qwen3.5 reasons before answering unless told not to. WebLLM's template doesn't know about
// thinking; `enable_thinking: false` makes WebLLM add the empty <think> block itself.
const QWEN35_FORMAT: ModelSpec['promptFormat'] = { kind: 'chatml', stopTokens: CHATML_STOPS, disableThinking: true, trimContent: true };
/** For models with WebLLM builds only. */
const WEBLLM_TEMPLATE: ModelSpec['promptFormat'] = { kind: 'model-template', stopTokens: [] };
const WEBLLM_TEMPLATE_NO_THINKING: ModelSpec['promptFormat'] = { ...WEBLLM_TEMPLATE, disableThinking: true };

// Models added with WebLLM (2026-10-10). Most also have ONNX files; those without a usable web
// export (Gemma 2, Qwen2.5 3B, and the 7B+ models) have nothing to fall back to, so on a device
// where WebLLM fails it is tried again each time. (Gemma 3 1B was dropped: WebLLM can't run its
// mixed local/global attention; see tools/webllm/make_catalogue.py.)


export const LLAMA32_1B_INSTRUCT: ModelSpec = {
  id: 'llama-3.2-1b-instruct',
  name: 'Llama 3.2 1B',
  publisher: 'Meta',
  parameters: '1.2B',
  license: 'Llama 3.2 Community License',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Llama-3.2-1B-Instruct-ONNX', revision: '14007543b6dc92de88daf96a9aa85d2f95ace6ef' },
  architecture: {
    family: 'llama',
    parameterCount: 1_236_000_000,
    layers: 16,
    hiddenSize: 2048,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 128256,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 11574638, sha256: '3a223ade375cc1d13b04e897ce1d36a04f50140e1ba3d107021ea68d4b5e614c' },
  promptFormat: LLAMA3_FORMAT,
  // Model card / generation_config: temperature 0.6, top-p 0.9, no repetition penalty.
  generation: { maxNewTokens: 1024, temperature: 0.6, topP: 0.9, repetitionPenalty: 1, repetitionWindow: 64, contextTokens: 4096 },
  quality: {
    rank: 3,
    summary: 'Meta’s small model: solid general knowledge and instruction following for its size; long context.',
    scores: [
      { benchmark: 'IFEval', score: 59.5, source: LLAMA32_CARD },
      { benchmark: 'MMLU', score: 49.3, source: LLAMA32_CARD },
      { benchmark: 'GSM8K (CoT)', score: 44.4, source: LLAMA32_CARD },
    ],
  },
  artifacts: [
    webllm('Llama-3.2-1B-Instruct-q4f16_1-MLC'),
    webllm('Llama-3.2-1B-Instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 149790, sha256: 'fccc573530dbefac92a36797262e4228dc6fb474734d80f6781e37744550333c' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 1089605632, sha256: '3f11fd184cd0b34f359cc2696c898026dfc0e169e56c3a2505ae964948a873f1' }],
      kvCacheType: 'float16',
      largestTensorBytes: 525336576,
      computeOps: 210,
    },
  ],
};

export const QWEN35_08B: ModelSpec = {
  id: 'qwen3.5-0.8b',
  name: 'Qwen3.5 0.8B',
  publisher: 'Alibaba Qwen',
  parameters: '0.8B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen3.5-0.8B-ONNX', revision: 'd1ea95774d40a29f0479bf9138cdf488a0394b6b' },
  // Hybrid attention: most layers use linear attention with a fixed-size state, so its
  // conversation memory is far smaller than this full-attention estimate.
  architecture: {
    family: 'qwen3.5',
    parameterCount: 800_000_000,
    layers: 24,
    hiddenSize: 1024,
    kvHeads: 2,
    headDim: 256,
    vocabSize: 248320,
    maxContextTokens: 262144,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 19226111, sha256: '89da80cc6689bef4d90cc1028249436975ffb0814618f1d93c65310e05801a9b' },
  promptFormat: QWEN35_FORMAT,
  generation: QWEN3_06B.generation,
  quality: {
    rank: 2,
    summary: 'The newest small Qwen: more general knowledge than Qwen3 0.6B, similar at following instructions.',
    scores: [
      { benchmark: 'IFEval (non-thinking)', score: 52.1, source: QWEN35_CARD },
      { benchmark: 'MMLU-Redux (non-thinking)', score: 48.5, source: QWEN35_CARD },
      { benchmark: 'MMLU-Pro (non-thinking)', score: 29.7, source: QWEN35_CARD },
    ],
  },
  artifacts: [
    webllm('Qwen3.5-0.8B-q4f16_1-MLC'),
    webllm('Qwen3.5-0.8B-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/decoder_model_merged_q4f16.onnx', sizeBytes: 1036898, sha256: '34e17c8e2035919df86ab1f52b41999a1bd18ba96b49dba6ac8d340aae652006' },
      externalData: [{ path: 'onnx/decoder_model_merged_q4f16.onnx_data', sizeBytes: 436662272, sha256: '468cf83a51e81e27ffb4210268b1b09979e68dd128ad5fe347e5d08721cecc41' }],
      embeddings: { graph: { path: 'onnx/embed_tokens_q4f16.onnx', sizeBytes: 1064, sha256: '8218531ac44ae9978d50647f1d907c53c308f758514b992504238c77843c254d' }, externalData: [{ path: 'onnx/embed_tokens_q4f16.onnx_data', sizeBytes: 147005440, sha256: 'ec4a1f13ff942653b52000a7a0ec40504110d8be9a0ecab2da4d3063588ed563' }] },
      kvCacheType: 'float16',
      largestTensorBytes: 127139840,
      computeOps: 367,
    },
  ],
};

export const QWEN25_15B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-1.5b-instruct',
  name: 'Qwen2.5 1.5B',
  publisher: 'Alibaba Qwen',
  parameters: '1.5B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen2.5-1.5B-Instruct', revision: 'd54957e3af1be04546436e874a60fe386041b244' },
  architecture: {
    family: 'qwen2',
    parameterCount: 1_544_000_000,
    layers: 28,
    hiddenSize: 1536,
    kvHeads: 2,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 7031673, sha256: 'a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf' },
  promptFormat: { kind: 'chatml', defaultSystemPrompt: QWEN25_05B_INSTRUCT.promptFormat.defaultSystemPrompt, stopTokens: CHATML_STOPS },
  generation: QWEN25_05B_INSTRUCT.generation,
  quality: {
    rank: 3,
    summary: 'Good at maths for its size; middling at following instructions.',
    scores: [
      { benchmark: 'IFEval (strict prompt)', score: 42.5, source: QWEN25_BLOG },
      { benchmark: 'MMLU-Redux', score: 50.7, source: QWEN25_BLOG },
      { benchmark: 'GSM8K', score: 73.2, source: QWEN25_BLOG },
    ],
  },
  artifacts: [
    webllm('Qwen2.5-1.5B-Instruct-q4f16_1-MLC'),
    webllm('Qwen2.5-1.5B-Instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 1787566590, sha256: '70c24509f760fed9a8f50391165c46a87660197028a42c4547763f1c173ddea6' },
      kvCacheType: 'float32',
      largestTensorBytes: 933494784,
      computeOps: 1248,
    },
  ],
};

export const QWEN25_CODER_15B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-coder-1.5b-instruct',
  name: 'Qwen2.5 Coder 1.5B',
  publisher: 'Alibaba Qwen',
  parameters: '1.5B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen2.5-Coder-1.5B-Instruct', revision: '774cc908b6c2be23bc4e8d4e401e4f8b8fba7daf' },
  architecture: QWEN25_15B_INSTRUCT.architecture,
  tokenizer: { path: 'tokenizer.json', sizeBytes: 11421896, sha256: '9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa' },
  promptFormat: { kind: 'chatml', defaultSystemPrompt: QWEN25_05B_INSTRUCT.promptFormat.defaultSystemPrompt, stopTokens: CHATML_STOPS },
  generation: QWEN25_05B_INSTRUCT.generation,
  quality: {
    rank: 3,
    summary: 'For code: writes and explains code well for its size. Weaker than general models at everything else.',
    scores: [
      { benchmark: 'HumanEval', score: 70.7, source: QWEN25_CODER_REPORT },
      { benchmark: 'MBPP', score: 69.2, source: QWEN25_CODER_REPORT },
      { benchmark: 'LiveCodeBench', score: 6.1, source: QWEN25_CODER_REPORT },
    ],
  },
  artifacts: [
    webllm('Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC'),
    webllm('Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 1344237070, sha256: '0e33cbfde4be371039daad0032fb3782170442e6901a25d4deabe96117b89882' },
      kvCacheType: 'float16',
      largestTensorBytes: 466747392,
      computeOps: 510,
    },
  ],
};

export const GEMMA2_2B_IT: ModelSpec = {
  id: 'gemma-2-2b-it',
  name: 'Gemma 2 2B',
  publisher: 'Google',
  parameters: '2.6B',
  license: 'Gemma Terms of Use',
  task: 'chat',
  source: webllmSource('gemma-2-2b-it-q4f16_1-MLC'),
  // Alternates local (4K window) and global attention; at the 4K context WebLLM's
  // build allows, every layer sees the whole conversation, so full attention is exact.
  architecture: {
    family: 'gemma2',
    parameterCount: 2_614_000_000,
    layers: 26,
    hiddenSize: 2304,
    kvHeads: 4,
    headDim: 256,
    vocabSize: 256000,
    maxContextTokens: 4096,
  },
  tokenizer: webllmTokenizer('gemma-2-2b-it-q4f16_1-MLC'),
  promptFormat: WEBLLM_TEMPLATE,
  // No published sampling settings; WebLLM's build defaults.
  generation: { maxNewTokens: 1024, temperature: 0.7, topP: 0.9, repetitionPenalty: 1, repetitionWindow: 64, contextTokens: 4096 },
  quality: {
    rank: 4,
    summary: 'Google’s small model: natural, well-written chat replies; weaker at maths. 4K context at most.',
    scores: [
      { benchmark: 'MMLU', score: 56.1, source: GEMMA2_REPORT },
      { benchmark: 'Chatbot Arena Elo', score: 1126, source: GEMMA2_REPORT },
      { benchmark: 'GSM8K (pre-trained model)', score: 24.3, source: GEMMA2_REPORT },
    ],
  },
  artifacts: [webllm('gemma-2-2b-it-q4f16_1-MLC'), webllm('gemma-2-2b-it-q4f32_1-MLC')],
};

export const QWEN3_17B: ModelSpec = {
  id: 'qwen3-1.7b',
  name: 'Qwen3 1.7B',
  publisher: 'Alibaba Qwen',
  parameters: '1.7B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen3-1.7B-ONNX', revision: 'b85539623180ae64303f37373d942cb0595ca44d' },
  architecture: {
    family: 'qwen3',
    parameterCount: 2_031_000_000,
    layers: 28,
    hiddenSize: 2048,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 40960,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 9117040, sha256: 'e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5' },
  promptFormat: QWEN3_06B.promptFormat,
  generation: QWEN3_06B.generation,
  quality: {
    rank: 5,
    summary: 'Clearly better reasoning, maths and instruction following than Qwen3 0.6B. A 1 GB download.',
    scores: [
      { benchmark: 'IFEval (strict prompt)', score: 68.2, source: QWEN3_REPORT },
      { benchmark: 'MMLU-Redux', score: 64.4, source: QWEN3_REPORT },
      { benchmark: 'MATH-500', score: 73.0, source: QWEN3_REPORT },
    ],
  },
  artifacts: [
    webllm('Qwen3-1.7B-q4f16_1-MLC'),
    webllm('Qwen3-1.7B-q4f32_1-MLC'),
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 2147212861, sha256: '3763020b62164be3d9c31dc6b15aa042052f5a9da846e991f6c45e0f1a6dced7' },
      kvCacheType: 'float32',
      largestTensorBytes: 1244659712,
      computeOps: 478,
    },
  ],
};

export const QWEN35_2B: ModelSpec = {
  id: 'qwen3.5-2b',
  name: 'Qwen3.5 2B',
  publisher: 'Alibaba Qwen',
  parameters: '2B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen3.5-2B-ONNX', revision: 'e988a4ce430d7dc6fd77b4b93341b0aeae289748' },
  architecture: {
    ...QWEN35_08B.architecture,
    parameterCount: 2_000_000_000,
    hiddenSize: 2048,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 19226111, sha256: '89da80cc6689bef4d90cc1028249436975ffb0814618f1d93c65310e05801a9b' },
  promptFormat: QWEN35_FORMAT,
  generation: QWEN3_06B.generation,
  quality: {
    rank: 5,
    summary: 'The newest Qwen at this size: more knowledge than Qwen3 1.7B, a little weaker at following instructions.',
    scores: [
      { benchmark: 'IFEval (non-thinking)', score: 61.2, source: QWEN35_CARD },
      { benchmark: 'MMLU-Redux (non-thinking)', score: 69.2, source: QWEN35_CARD },
      { benchmark: 'MMLU-Pro (non-thinking)', score: 55.3, source: QWEN35_CARD },
    ],
  },
  artifacts: [
    webllm('Qwen3.5-2B-q4f16_1-MLC'),
    webllm('Qwen3.5-2B-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/decoder_model_merged_q4f16.onnx', sizeBytes: 1046438, sha256: 'aff64289c18b527ee16963444be405d08e7211c4fdca6f3fe53d9d4586703bf7' },
      externalData: [{ path: 'onnx/decoder_model_merged_q4f16.onnx_data', sizeBytes: 1089777664, sha256: 'e0d1375a8d30aa7db927ce05c60c4134345dca61150acdf05b9822bc06c7e54b' }],
      embeddings: { graph: { path: 'onnx/embed_tokens_q4f16.onnx', sizeBytes: 1064, sha256: '802a072ff21f540eda7f343aa71dbb0354c8859caaf34f09b3bf8117725d7de8' }, externalData: [{ path: 'onnx/embed_tokens_q4f16.onnx_data', sizeBytes: 294010880, sha256: '650aa8eb39b7404ca2c908d78243c82b6fd88321feeb8fca175745806c6b3a81' }] },
      kvCacheType: 'float16',
      largestTensorBytes: 254279680,
      computeOps: 367,
    },
  ],
};

export const QWEN25_3B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-3b-instruct',
  name: 'Qwen2.5 3B',
  publisher: 'Alibaba Qwen',
  parameters: '3B',
  license: 'Qwen Research License (non-commercial)',
  task: 'chat',
  source: webllmSource('Qwen2.5-3B-Instruct-q4f16_1-MLC'),
  architecture: {
    family: 'qwen2',
    parameterCount: 3_086_000_000,
    layers: 36,
    hiddenSize: 2048,
    kvHeads: 2,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 32768,
  },
  tokenizer: webllmTokenizer('Qwen2.5-3B-Instruct-q4f16_1-MLC'),
  promptFormat: WEBLLM_TEMPLATE,
  generation: QWEN25_05B_INSTRUCT.generation,
  quality: {
    rank: 5,
    summary: 'Strong at maths and code for its size. Its licence doesn’t allow commercial use.',
    scores: [
      { benchmark: 'IFEval (strict prompt)', score: 58.2, source: QWEN25_BLOG },
      { benchmark: 'MMLU-Redux', score: 64.4, source: QWEN25_BLOG },
      { benchmark: 'GSM8K', score: 86.7, source: QWEN25_BLOG },
    ],
  },
  artifacts: [webllm('Qwen2.5-3B-Instruct-q4f16_1-MLC'), webllm('Qwen2.5-3B-Instruct-q4f32_1-MLC')],
};

export const LLAMA32_3B_INSTRUCT: ModelSpec = {
  id: 'llama-3.2-3b-instruct',
  name: 'Llama 3.2 3B',
  publisher: 'Meta',
  parameters: '3.2B',
  license: 'Llama 3.2 Community License',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Llama-3.2-3B-Instruct-ONNX', revision: 'cab364e7d0e1de7aa09e3abc932be92361c5b55f' },
  architecture: {
    family: 'llama',
    parameterCount: 3_213_000_000,
    layers: 28,
    hiddenSize: 3072,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 128256,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 11574638, sha256: '3a223ade375cc1d13b04e897ce1d36a04f50140e1ba3d107021ea68d4b5e614c' },
  promptFormat: LLAMA3_FORMAT,
  generation: LLAMA32_1B_INSTRUCT.generation,
  quality: {
    rank: 6,
    summary: 'Very good at following instructions; solid knowledge and maths. A 1.8 GB download.',
    scores: [
      { benchmark: 'IFEval', score: 77.4, source: LLAMA32_CARD },
      { benchmark: 'MMLU', score: 63.4, source: LLAMA32_CARD },
      { benchmark: 'GSM8K (CoT)', score: 77.7, source: LLAMA32_CARD },
    ],
  },
  artifacts: [
    webllm('Llama-3.2-3B-Instruct-q4f16_1-MLC'),
    webllm('Llama-3.2-3B-Instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 260899, sha256: '43648be8ff45ed7bc75c75ea0d495beffa8a8632910e53d3aa824d6bfffaae46' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2095929344, sha256: '0669c8c258ea5437b82cc17e5ca87bb91a9ede5b2f5ff80675c0b8e51f1b6043' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 311427072, sha256: '63b1b82298ad66f940b4f918f81c386fbe4e15a4e178efb14bc558d127185113' }],
      kvCacheType: 'float16',
      largestTensorBytes: 788004864,
      computeOps: 366,
    },
  ],
};

// Phi models: no published sampling settings; moderate ones, no repetition penalty.
const PHI_GENERATION: GenerationDefaults = {
  maxNewTokens: 1024,
  temperature: 0.7,
  topP: 0.9,
  repetitionPenalty: 1,
  repetitionWindow: 64,
  contextTokens: 4096,
};

export const PHI35_MINI_INSTRUCT: ModelSpec = {
  id: 'phi-3.5-mini-instruct',
  name: 'Phi-3.5 mini',
  publisher: 'Microsoft',
  parameters: '3.8B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Phi-3.5-mini-instruct-onnx-web', revision: '1a2c0e4b2eb1a872716fdc50b100617499739a73' },
  // No grouped attention (32 KV heads): its conversation memory is ~1.5 GB at 4K context.
  architecture: {
    family: 'phi3',
    parameterCount: 3_821_000_000,
    layers: 32,
    hiddenSize: 3072,
    kvHeads: 32,
    headDim: 96,
    vocabSize: 32064,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 1844408, sha256: '9e2ae3d66819f163cdcfedba5078f3a3af8118c2712491ed60912f77886bda6f' },
  promptFormat: PHI_FORMAT,
  generation: PHI_GENERATION,
  quality: {
    rank: 6,
    summary: 'Broad knowledge and good reasoning, mostly in English. Needs a lot of GPU memory (about 3.7 GB).',
    scores: [
      { benchmark: 'MMLU (5-shot)', score: 69.0, source: PHI35_CARD },
      { benchmark: 'MMLU-Pro (CoT)', score: 47.4, source: PHI35_CARD },
      { benchmark: 'GSM8K (8-shot, CoT)', score: 86.2, source: PHI35_CARD },
    ],
  },
  artifacts: [
    webllm('Phi-3.5-mini-instruct-q4f16_1-MLC'),
    webllm('Phi-3.5-mini-instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 220452346, sha256: '1e1119b8d3fa1adaaad4abaf0c0405dec377054948440e02dd6837d18d2a920d' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2097020928, sha256: 'b6dfb3069b717b518266dd58cb8e794bc0712fbc46e1737aef99cd8875302cf0' }],
      kvCacheType: 'float16',
      largestTensorBytes: 197001216,
      computeOps: 488,
    },
  ],
};

export const PHI4_MINI_INSTRUCT: ModelSpec = {
  id: 'phi-4-mini-instruct',
  name: 'Phi-4 mini',
  publisher: 'Microsoft',
  parameters: '3.8B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Phi-4-mini-instruct-ONNX', revision: 'e61f45fc5fabba2aee31ff85ba4cf99219b4bf28' },
  architecture: {
    family: 'phi3',
    parameterCount: 3_836_000_000,
    layers: 32,
    hiddenSize: 3072,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 200064,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 13303196, sha256: '9ca5aa723a31a7a122497e059bd48dd67a5bd03ad16b3ffcf16093fd3021c1eb' },
  promptFormat: PHI_FORMAT,
  generation: PHI_GENERATION,
  quality: {
    rank: 7,
    summary: 'Strong at maths and reasoning, with broad knowledge. A 2.2 GB download.',
    scores: [
      { benchmark: 'MMLU (5-shot)', score: 67.3, source: PHI4_CARD },
      { benchmark: 'MMLU-Pro (CoT)', score: 52.8, source: PHI4_CARD },
      { benchmark: 'MATH (CoT)', score: 64.0, source: PHI4_CARD },
    ],
  },
  artifacts: [
    webllm('Phi-4-mini-instruct-q4f16_1-MLC'),
    webllm('Phi-4-mini-instruct-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 26270832, sha256: 'ca26127777adf1df99b5fc1a3b4d1e0c426a6bf56626889873bfc4a6a095b4fc' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2087043072, sha256: '385526d648e4b3e361f3117564a6bd3cad7712a5c06fa23401763187674fd46c' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 438239232, sha256: 'b9a5d6f40fde30e9155d671dc630d2ea554f903f01c4ef8bbdc64c5b38035923' }],
      kvCacheType: 'float16',
      largestTensorBytes: 307298304,
      computeOps: 419,
    },
  ],
};

export const QWEN3_4B: ModelSpec = {
  id: 'qwen3-4b',
  name: 'Qwen3 4B',
  publisher: 'Alibaba Qwen',
  parameters: '4B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen3-4B-ONNX', revision: '5cbfbcacd84ed9e54cea017a4cbddb67cc45337f' },
  architecture: {
    family: 'qwen3',
    parameterCount: 4_022_000_000,
    layers: 36,
    hiddenSize: 2560,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 40960,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 9117040, sha256: 'e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5' },
  promptFormat: QWEN3_06B.promptFormat,
  generation: QWEN3_06B.generation,
  quality: {
    rank: 8,
    summary: 'Best at reasoning, maths and following instructions below the large models. Needs a capable GPU; a 2.3 GB download.',
    scores: [
      { benchmark: 'IFEval (strict prompt)', score: 81.2, source: QWEN3_REPORT_4B },
      { benchmark: 'MMLU-Redux', score: 77.3, source: QWEN3_REPORT_4B },
      { benchmark: 'MATH-500', score: 84.8, source: QWEN3_REPORT_4B },
    ],
  },
  artifacts: [
    webllm('Qwen3-4B-q4f16_1-MLC'),
    webllm('Qwen3-4B-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 59762833, sha256: 'd3e946f9e38577411b0251051c91a1f20c3c0c831e1cf76a13c58ce279d950de' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2096005120, sha256: '050398248de4fce7b31b2d2caa909596e4c7aa0f35696270df46b8aaf2209fc8' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 677150720, sha256: '363ff5e70ebeb5866afea8eb80b7bdfe22d94d735a8c4d2ebf90c75424c2a410' }],
      kvCacheType: 'float16',
      largestTensorBytes: 777912320,
      computeOps: 614,
    },
  ],
};

// Large tier: 4-9B, 3.9-6.4 GB of GPU memory with the fp16 builds. For desktops and
// laptops with plenty of memory; the recommendations mark them "may not fit" elsewhere.

export const QWEN35_4B: ModelSpec = {
  id: 'qwen3.5-4b',
  name: 'Qwen3.5 4B',
  publisher: 'Alibaba Qwen',
  parameters: '4B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen3.5-4B-ONNX', revision: '3551704c0c04ee52b5f1bfc27d1ee8d2a94973fc' },
  architecture: { ...QWEN35_08B.architecture, parameterCount: 4_000_000_000, layers: 32, hiddenSize: 2560, kvHeads: 4 },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 19226111, sha256: '89da80cc6689bef4d90cc1028249436975ffb0814618f1d93c65310e05801a9b' },
  promptFormat: QWEN35_FORMAT,
  generation: QWEN3_06B.generation,
  quality: {
    rank: 8,
    // The card publishes thinking-mode scores only; the app answers without thinking, which scores lower.
    summary: 'The newest Qwen at 4B: strong knowledge and instruction following. Needs about 4 GB of GPU memory.',
    scores: [
      { benchmark: 'IFEval (thinking)', score: 89.8, source: QWEN35_CARD },
      { benchmark: 'MMLU-Redux (thinking)', score: 88.8, source: QWEN35_CARD },
      { benchmark: 'MMLU-Pro (thinking)', score: 79.1, source: QWEN35_CARD },
    ],
  },
  artifacts: [
    webllm('Qwen3.5-4B-q4f16_1-MLC'),
    webllm('Qwen3.5-4B-q4f32_1-MLC'),
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/decoder_model_merged_q4f16.onnx', sizeBytes: 1421099, sha256: '61b653b72f301d1ff503910d23f67ee062e41195257c874e2ef5b73e36cc00b5' },
      externalData: [{ path: 'onnx/decoder_model_merged_q4f16.onnx_data', sizeBytes: 2067208192, sha256: '846fad79745d84482f8baba2d5db2d907ce42704c1b0323a3467a52288c4165b' }, { path: 'onnx/decoder_model_merged_q4f16.onnx_data_1', sizeBytes: 367513600, sha256: 'fc1bb145d8839272a87c71e0cb4d34832a0d7bb4de06ab4fb74fea1aa6ddf7e5' }],
      embeddings: { graph: { path: 'onnx/embed_tokens_q4f16.onnx', sizeBytes: 1064, sha256: '0e5fe965e5575b6428b7dea82661ed09bf7abadf29450c279e46e8113745110e' }, externalData: [{ path: 'onnx/embed_tokens_q4f16.onnx_data', sizeBytes: 367513600, sha256: 'fc1bb145d8839272a87c71e0cb4d34832a0d7bb4de06ab4fb74fea1aa6ddf7e5' }] },
      kvCacheType: 'float16',
      largestTensorBytes: 317849600,
      computeOps: 487,
    },
  ],
};

export const QWEN25_CODER_7B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-coder-7b-instruct',
  name: 'Qwen2.5 Coder 7B',
  publisher: 'Alibaba Qwen',
  parameters: '7.6B',
  license: 'Apache-2.0',
  task: 'chat',
  source: webllmSource('Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC'),
  architecture: {
    family: 'qwen2',
    parameterCount: 7_616_000_000,
    layers: 28,
    hiddenSize: 3584,
    kvHeads: 4,
    headDim: 128,
    vocabSize: 152064,
    maxContextTokens: 32768,
  },
  tokenizer: webllmTokenizer('Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC'),
  promptFormat: WEBLLM_TEMPLATE,
  generation: QWEN25_05B_INSTRUCT.generation,
  quality: {
    rank: 8,
    summary: 'For code: much stronger than the 1.5B coder. Needs about 5 GB of GPU memory; a 4.3 GB download.',
    scores: [
      { benchmark: 'HumanEval', score: 88.4, source: QWEN25_CODER_REPORT },
      { benchmark: 'MBPP', score: 83.5, source: QWEN25_CODER_REPORT },
      { benchmark: 'LiveCodeBench', score: 18.2, source: QWEN25_CODER_REPORT },
    ],
  },
  artifacts: [webllm('Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC'), webllm('Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC')],
};

export const LLAMA31_8B_INSTRUCT: ModelSpec = {
  id: 'llama-3.1-8b-instruct',
  name: 'Llama 3.1 8B',
  publisher: 'Meta',
  parameters: '8B',
  license: 'Llama 3.1 Community License',
  task: 'chat',
  source: webllmSource('Llama-3.1-8B-Instruct-q4f16_1-MLC'),
  architecture: {
    family: 'llama',
    parameterCount: 8_030_000_000,
    layers: 32,
    hiddenSize: 4096,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 128256,
    maxContextTokens: 131072,
  },
  tokenizer: webllmTokenizer('Llama-3.1-8B-Instruct-q4f16_1-MLC'),
  promptFormat: WEBLLM_TEMPLATE,
  generation: LLAMA32_1B_INSTRUCT.generation,
  quality: {
    rank: 9,
    summary: 'Meta’s well-rounded 8B model: very good at following instructions, broad knowledge. Needs about 5 GB of GPU memory.',
    scores: [
      { benchmark: 'IFEval', score: 80.4, source: LLAMA31_CARD },
      { benchmark: 'MMLU (0-shot, CoT)', score: 73.0, source: LLAMA31_CARD },
      { benchmark: 'GSM8K (8-shot, CoT)', score: 84.5, source: LLAMA31_CARD },
    ],
  },
  artifacts: [webllm('Llama-3.1-8B-Instruct-q4f16_1-MLC'), webllm('Llama-3.1-8B-Instruct-q4f32_1-MLC')],
};

export const GEMMA2_9B_IT: ModelSpec = {
  id: 'gemma-2-9b-it',
  name: 'Gemma 2 9B',
  publisher: 'Google',
  parameters: '9B',
  license: 'Gemma Terms of Use',
  task: 'chat',
  source: webllmSource('gemma-2-9b-it-q4f16_1-MLC'),
  architecture: {
    family: 'gemma2',
    parameterCount: 9_242_000_000,
    layers: 42,
    hiddenSize: 3584,
    kvHeads: 8,
    headDim: 256,
    vocabSize: 256000,
    maxContextTokens: 4096,
  },
  tokenizer: webllmTokenizer('gemma-2-9b-it-q4f16_1-MLC'),
  promptFormat: GEMMA2_2B_IT.promptFormat,
  generation: GEMMA2_2B_IT.generation,
  quality: {
    rank: 9,
    summary: 'Google’s 9B model: natural, well-written replies. 4K context at most; needs about 6.4 GB of GPU memory.',
    scores: [
      { benchmark: 'MMLU', score: 72.3, source: GEMMA2_REPORT },
      { benchmark: 'Chatbot Arena Elo', score: 1187, source: GEMMA2_REPORT },
      { benchmark: 'GSM8K (pre-trained model)', score: 68.6, source: GEMMA2_REPORT },
    ],
  },
  artifacts: [webllm('gemma-2-9b-it-q4f16_1-MLC'), webllm('gemma-2-9b-it-q4f32_1-MLC')],
};

export const QWEN3_8B: ModelSpec = {
  id: 'qwen3-8b',
  name: 'Qwen3 8B',
  publisher: 'Alibaba Qwen',
  parameters: '8B',
  license: 'Apache-2.0',
  task: 'chat',
  source: webllmSource('Qwen3-8B-q4f16_1-MLC'),
  architecture: { ...QWEN3_4B.architecture, parameterCount: 8_191_000_000, hiddenSize: 4096 },
  tokenizer: webllmTokenizer('Qwen3-8B-q4f16_1-MLC'),
  promptFormat: WEBLLM_TEMPLATE_NO_THINKING,
  generation: QWEN3_06B.generation,
  quality: {
    rank: 10,
    summary: 'Excellent reasoning, maths and instruction following. Needs about 5.7 GB of GPU memory; a 4.6 GB download.',
    scores: [
      { benchmark: 'IFEval (strict prompt)', score: 83.0, source: QWEN3_REPORT_4B },
      { benchmark: 'MMLU-Redux', score: 79.5, source: QWEN3_REPORT_4B },
      { benchmark: 'MATH-500', score: 87.4, source: QWEN3_REPORT_4B },
    ],
  },
  artifacts: [webllm('Qwen3-8B-q4f16_1-MLC'), webllm('Qwen3-8B-q4f32_1-MLC')],
};

export const QWEN35_9B: ModelSpec = {
  id: 'qwen3.5-9b',
  name: 'Qwen3.5 9B',
  publisher: 'Alibaba Qwen',
  parameters: '9B',
  license: 'Apache-2.0',
  task: 'chat',
  source: webllmSource('Qwen3.5-9B-q4f16_1-MLC'),
  architecture: { ...QWEN35_4B.architecture, parameterCount: 9_000_000_000, hiddenSize: 4096 },
  tokenizer: webllmTokenizer('Qwen3.5-9B-q4f16_1-MLC'),
  promptFormat: WEBLLM_TEMPLATE_NO_THINKING,
  generation: QWEN3_06B.generation,
  quality: {
    rank: 10,
    summary: 'The strongest here: the newest Qwen, best at knowledge and instructions. Needs about 6.4 GB of GPU memory.',
    scores: [
      { benchmark: 'IFEval (thinking)', score: 91.5, source: QWEN35_CARD },
      { benchmark: 'MMLU-Redux (thinking)', score: 91.1, source: QWEN35_CARD },
      { benchmark: 'MMLU-Pro (thinking)', score: 82.5, source: QWEN35_CARD },
    ],
  },
  artifacts: [webllm('Qwen3.5-9B-q4f16_1-MLC'), webllm('Qwen3.5-9B-q4f32_1-MLC')],
};

export const LLAMA31_70B_INSTRUCT: ModelSpec = {
  id: 'llama-3.1-70b-instruct',
  name: 'Llama 3.1 70B',
  publisher: 'Meta',
  parameters: '70B',
  license: 'Llama 3.1 Community License',
  task: 'chat',
  source: webllmSource('Llama-3.1-70B-Instruct-q3f16_1-MLC'),
  architecture: {
    family: 'llama',
    parameterCount: 70_554_000_000,
    layers: 80,
    hiddenSize: 8192,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 128256,
    maxContextTokens: 131072,
  },
  tokenizer: webllmTokenizer('Llama-3.1-70B-Instruct-q3f16_1-MLC'),
  promptFormat: WEBLLM_TEMPLATE,
  generation: LLAMA32_1B_INSTRUCT.generation,
  quality: {
    rank: 12,
    summary: 'A far larger model: much better knowledge, reasoning and writing. 3-bit, a 32 GB download; slow even on big machines.',
    scores: [
      { benchmark: 'IFEval', score: 87.5, source: LLAMA31_CARD },
      { benchmark: 'MMLU (0-shot, CoT)', score: 86.0, source: LLAMA31_CARD },
      { benchmark: 'GSM8K (8-shot, CoT)', score: 95.1, source: LLAMA31_CARD },
    ],
  },
  experimental:
    'Untested: it needs about 31 GB of GPU memory (a Mac with 48 GB or more, or a GPU with 32 GB), more than any machine we test on.',
  artifacts: [webllm('Llama-3.1-70B-Instruct-q3f16_1-MLC')],
};

/** Smallest first, roughly. */
export const catalogue: readonly ModelSpec[] = [
  SMOLLM2_360M_INSTRUCT,
  QWEN25_05B_INSTRUCT,
  QWEN3_06B,
  QWEN35_08B,
  LLAMA32_1B_INSTRUCT,
  QWEN25_15B_INSTRUCT,
  QWEN25_CODER_15B_INSTRUCT,
  SMOLLM2_17B_INSTRUCT,
  QWEN3_17B,
  QWEN35_2B,
  GEMMA2_2B_IT,
  QWEN25_3B_INSTRUCT,
  LLAMA32_3B_INSTRUCT,
  PHI35_MINI_INSTRUCT,
  PHI4_MINI_INSTRUCT,
  QWEN3_4B,
  QWEN35_4B,
  QWEN25_CODER_7B_INSTRUCT,
  LLAMA31_8B_INSTRUCT,
  QWEN3_8B,
  GEMMA2_9B_IT,
  QWEN35_9B,
  LLAMA31_70B_INSTRUCT,
];

export function findModel(id: string): ModelSpec | undefined {
  return catalogue.find((m) => m.id === id);
}

/** The files the ONNX engine's load of `artifact` needs, tokenizer first (small, so network problems show early) and the graph last. */
export function artifactFiles(model: ModelSpec, artifact: OnnxArtifact): ArtifactFile[] {
  return [model.tokenizer, ...onnxWeightFiles(artifact)];
}

/**
 * The ONNX files holding the weights, which are read tensor by tensor from disk
 * rather than whole: the embedding graph and its data (if separate), the
 * graph's external data, then the graph.
 */
export function onnxWeightFiles(artifact: OnnxArtifact): ArtifactFile[] {
  const embeddings = artifact.embeddings ? [artifact.embeddings.graph, ...artifact.embeddings.externalData] : [];
  return [...embeddings, ...(artifact.externalData ?? []), artifact.graph];
}

/** The engine's name as shown to people. */
export function engineName(engine: EngineId): string {
  return engine === 'webllm' ? 'WebLLM' : 'ONNX Runtime';
}

/** Total download size of an artifact (with the tokenizer for ONNX). */
export function artifactBytes(model: ModelSpec, artifact: ModelArtifact): number {
  if (artifact.engine === 'webllm') {
    return artifact.build.downloadBytes;
  }
  return model.tokenizer.sizeBytes + weightBytes(artifact);
}

/** Bytes of model weights (what each generated token reads). */
export function weightBytes(artifact: ModelArtifact): number {
  if (artifact.engine === 'webllm') {
    return artifact.build.weightBytes;
  }
  return onnxWeightFiles(artifact).reduce((n, f) => n + f.sizeBytes, 0);
}

/**
 * Identifies exactly which files an artifact runs, so measurements of one file
 * are never applied to another: the graph's SHA-256, or a WebLLM build's
 * repository commit.
 */
export function artifactFingerprint(artifact: ModelArtifact): string {
  return artifact.engine === 'webllm' ? `${artifact.build.repo}@${artifact.build.revision}` : artifact.graph.sha256;
}

/**
 * A WebLLM artifact from a pinned build. The fp16 builds always need
 * `shader-f16` (WebLLM's own list omits it for some models).
 */
function webllm(modelId: keyof typeof WEBLLM_BUILDS): WebLlmArtifact {
  const precision: Precision = modelId.includes('q4f16') ? 'q4f16' : modelId.includes('q3f16') ? 'q3f16' : 'q4';
  return {
    id: `webllm-${precision}`,
    engine: 'webllm',
    format: 'mlc',
    precision,
    requiredGpuFeatures: precision === 'q4' ? [] : ['shader-f16'],
    modelId,
    build: WEBLLM_BUILDS[modelId],
  };
}

/** `source` of a model with only WebLLM builds: its main build's repository. */
function webllmSource(modelId: keyof typeof WEBLLM_BUILDS): ModelSpec['source'] {
  const { repo, revision } = WEBLLM_BUILDS[modelId];
  return { host: 'huggingface', repo, revision };
}

/** `tokenizer` of a model with only WebLLM builds: its main build's tokenizer file. */
function webllmTokenizer(modelId: keyof typeof WEBLLM_BUILDS): ArtifactFile {
  const { path, sizeBytes, sha256 } = WEBLLM_BUILDS[modelId].tokenizer;
  if (!sha256) throw new Error(`No tokenizer hash in WebLLM build ${modelId}`);
  return { path, sizeBytes, sha256 };
}

export function artifactUrl(model: ModelSpec, file: ArtifactFile): string {
  const { repo, revision } = model.source;
  return `https://huggingface.co/${repo}/resolve/${revision}/${file.path}`;
}
