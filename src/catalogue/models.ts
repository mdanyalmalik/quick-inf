// Model catalogue: stable model identities, separate from the artifacts that
// execute them. A model can have several artifacts (precisions), listed in
// order of preference; the runtime picks the first this device can run (see
// src/runtime/compatibility.ts). Every model runs on this app's own ONNX
// Runtime engine (WebLLM was removed on 2026-10-11, see info/devlog.md).
//
// ONNX artifact entries are printed by tools/reference/onnx_artifact.py from the
// files tools/reference/fetch_model.py pins; the original models are checked
// against their references (tools/reference/make_fixtures.py).

import type { PromptFormatKind } from '../tokenizer/prompt-format';

/** 4-bit weights with fp16 maths ('q4f16') or fp32 maths ('q4', for GPUs without fp16 shaders). */
export type Precision = 'q4f16' | 'q4' | 'fp16';

/** The runtime that runs an artifact: this app's own ONNX Runtime adapter (src/runtime/onnx). */
export type EngineId = 'onnx-webgpu';

export interface ArtifactFile {
  /** Path within the pinned repository revision. */
  path: string;
  sizeBytes: number;
  sha256: string;
  /** Downloaded from another repository (saved with the model under `path`). */
  from?: { repo: string; revision: string; path: string };
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

export type ModelArtifact = OnnxArtifact;

export interface ModelArchitecture {
  family: 'llama' | 'qwen2' | 'qwen3' | 'qwen3.5' | 'phi3' | 'gemma2' | 'gemma3' | 'lfm2' | 'granite' | 'olmo3';
  /** Total parameters, including embeddings (used for compute estimates). */
  parameterCount: number;
  /** Mixture-of-experts models: the parameters each token uses. */
  activeParameterCount?: number;
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
   * by.
   */
  source: { host: 'huggingface'; repo: string; revision: string };
  architecture: ModelArchitecture;
  /** tokenizer.json in `source`. */
  tokenizer: ArtifactFile;
  promptFormat: {
    /** The format this app builds (see tokenizer/prompt-format.ts). */
    kind: PromptFormatKind;
    defaultSystemPrompt?: string;
    stopTokens: string[];
    /** Qwen3: answer without a reasoning phase (the template's enable_thinking=false). */
    disableThinking?: boolean;
    /** The template trims each message (Qwen3.5). */
    trimContent?: boolean;
    /** The control token every prompt starts with (LFM2, Phi-3). */
    beginToken?: string;
    /** phi: the template drops system messages (Phi-3). */
    noSystem?: boolean;
    /** phi: text every system turn starts with (Phi-4-mini-reasoning). */
    systemPrefix?: string;
    /** phi4: the system turn every prompt has, in place of the conversation's (Phi-4-reasoning). */
    fixedSystemPrompt?: string;
    /**
     * The model thinks before answering, between <think> and </think> (shown apart from the
     * answer, and left out of the conversation history the model sees next time).
     */
    reasoning?: boolean;
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
const PHI35_CARD = 'Phi-3.5-mini-instruct model card (Microsoft)';
const PHI4_CARD = 'Phi-4-mini-instruct model card (Microsoft)';
const PHI3_CARD = 'Phi-3-mini-4k-instruct model card, June 2024 update (Microsoft)';
const GEMMA3_REPORT = 'Gemma 3 technical report (instruction-tuned models)';
const GEMMA3_270M_CARD = 'Gemma 3 270M model card (Google)';
const GRANITE4_CARD = 'Granite 4.0 model card (IBM)';
const LFM2_CARD = 'LFM2-700M model card (Liquid AI)';
const LFM2_26B_CARD = 'LFM2-2.6B model card (Liquid AI)';
const LFM2_8B_CARD = 'LFM2-8B-A1B model card (Liquid AI)';
const LFM25_350M_CARD = 'LFM2.5-350M model card (Liquid AI)';
const LFM25_12B_CARD = 'LFM2.5-1.2B-Instruct model card (Liquid AI)';
const OLMO3_CARD = 'Olmo 3 7B Instruct model card (Ai2)';
const LLAMA31_CARD = 'Llama 3.1 model card (Meta)';
const GEMMA2_REPORT = 'Gemma 2 technical report, tables 13, 14 and 17';
const DEEPSEEK_R1_REPORT = 'DeepSeek-R1 model card, distilled models (pass@1)';
const PHI4_MINI_REASONING_CARD = 'Phi-4-mini-reasoning model card (Microsoft)';
const PHI3_MEDIUM_CARD = 'Phi-3-medium-4k-instruct model card (Microsoft)';
const PHI4_REPORT = 'Phi-4 model card (Microsoft)';
const PHI4_REASONING_CARD = 'Phi-4-reasoning model card (Microsoft)';
const QWEN3_2507_CARD = 'Qwen3-4B-Instruct-2507 model card';
const QWEN25_MATH_REPORT = 'Qwen2.5-Math technical report (CoT, English)';
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
// Qwen3.5 reasons before answering unless told not to.
const QWEN35_FORMAT: ModelSpec['promptFormat'] = { kind: 'chatml', stopTokens: CHATML_STOPS, disableThinking: true, trimContent: true };

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

/** Smallest first, roughly. */
// Models added 2026-10-11, when WebLLM was removed: every web-ready ONNX export of a
// supported architecture and chat format that fits a 24 GB Mac, plus one that doesn't.

const GEMMA_FORMAT: ModelSpec['promptFormat'] = { kind: 'gemma', stopTokens: ['<end_of_turn>', '<eos>'] };
// Gemma's card samples at temperature 1.0 with top-k 64; without top-k, a lower temperature
// keeps the small models from wandering.
const GEMMA_GENERATION: GenerationDefaults = { maxNewTokens: 1024, temperature: 0.7, topP: 0.95, repetitionPenalty: 1, repetitionWindow: 64, contextTokens: 4096 };
const LFM_FORMAT: ModelSpec['promptFormat'] = { kind: 'chatml', beginToken: '<|startoftext|>', stopTokens: ['<|im_end|>'] };
// Liquid AI's recommended settings (temperature 0.3, repetition penalty 1.05; min-p 0.15 left out).
const LFM_GENERATION: GenerationDefaults = { maxNewTokens: 1024, temperature: 0.3, topP: 0.9, repetitionPenalty: 1.05, repetitionWindow: 64, contextTokens: 4096 };
const GRANITE_FORMAT: ModelSpec['promptFormat'] = {
  kind: 'granite',
  defaultSystemPrompt: 'You are a helpful assistant. Please ensure responses are professional, accurate, and safe.',
  stopTokens: ['<|end_of_text|>'],
};
// Granite 4.0 Micro's template has no default system prompt, unlike the smaller ones'.
const GRANITE_MICRO_FORMAT: ModelSpec['promptFormat'] = { kind: 'granite', stopTokens: ['<|end_of_text|>'] };
// Phi-3's template starts with <s> and leaves out system messages.
const PHI3_FORMAT: ModelSpec['promptFormat'] = { ...PHI_FORMAT, beginToken: '<s>', noSystem: true };
const OLMO3_FORMAT: ModelSpec['promptFormat'] = {
  kind: 'chatml',
  defaultSystemPrompt: 'You are a helpful function-calling AI assistant. You do not currently have access to any functions. <functions></functions>',
  stopTokens: ['<|im_end|>', '<|endoftext|>'],
};
// generation_config.json: temperature 0.6, top-p 0.95.
const OLMO3_GENERATION: GenerationDefaults = { maxNewTokens: 1024, temperature: 0.6, topP: 0.95, repetitionPenalty: 1, repetitionWindow: 64, contextTokens: 4096 };

// Thinking models: room for a long reasoning phase before the answer.
const REASONING_GENERATION: GenerationDefaults = { maxNewTokens: 4096, temperature: 0.6, topP: 0.95, repetitionPenalty: 1, repetitionWindow: 64, contextTokens: 8192 };
const DEEPSEEK_FORMAT: ModelSpec['promptFormat'] = { kind: 'deepseek', stopTokens: ['<｜end▁of▁sentence｜>'], reasoning: true };
const MISTRAL_FORMAT: ModelSpec['promptFormat'] = { kind: 'mistral', stopTokens: ['</s>'] };
const MISTRAL_GENERATION: GenerationDefaults = { maxNewTokens: 1024, temperature: 0.7, topP: 0.9, repetitionPenalty: 1, repetitionWindow: 64, contextTokens: 4096 };
// Phi-3 Medium's template has no <s> and leaves out system messages.
const PHI3_MEDIUM_FORMAT: ModelSpec['promptFormat'] = { ...PHI_FORMAT, noSystem: true };
const PHI4_MINI_REASONING_FORMAT: ModelSpec['promptFormat'] = {
  ...PHI_FORMAT,
  systemPrefix: 'Your name is Phi, an AI math expert developed by Microsoft.',
  reasoning: true,
};
const PHI4_FORMAT: ModelSpec['promptFormat'] = { kind: 'phi4', stopTokens: ['<|im_end|>', '<|endoftext|>'] };
const PHI4_REASONING_FORMAT: ModelSpec['promptFormat'] = {
  ...PHI4_FORMAT,
  fixedSystemPrompt:
    'You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:',
  reasoning: true,
};

// NEW_MODELS

export const DEEPSEEK_R1_QWEN_15B: ModelSpec = {
  id: 'deepseek-r1-distill-qwen-1.5b',
  name: 'DeepSeek-R1 Qwen 1.5B',
  publisher: 'DeepSeek',
  parameters: '1.5B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX', revision: 'd63c4c38c538da4272c6f9ef3274196a0853b2d0' },
  architecture: {
    family: 'qwen2',
    parameterCount: 1_777_088_000,
    layers: 28,
    hiddenSize: 1536,
    kvHeads: 2,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 9117072, sha256: '0d7259f8a225ceaa421baf5ae0687e0e4a4239b86a6d1838aba50e2470d99341' },
  promptFormat: DEEPSEEK_FORMAT,
  generation: REASONING_GENERATION,
  quality: {
    rank: 2,
    summary: 'A small model that thinks before answering: good at school maths, weak at everything else.',
    scores: [
      { benchmark: 'MATH-500', score: 83.9, source: DEEPSEEK_R1_REPORT },
      { benchmark: 'AIME 2024', score: 28.9, source: DEEPSEEK_R1_REPORT },
      { benchmark: 'GPQA Diamond', score: 33.8, source: DEEPSEEK_R1_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 1966462264, sha256: '3c64abf7c85102af29c0c0231caf754e981faebfa83ef9530fb457d2c5140b04' },
      kvCacheType: 'float32',
      largestTensorBytes: 933494784,
      computeOps: 510,
    },
  ],
};

export const QWEN3_4B_2507: ModelSpec = {
  id: 'qwen3-4b-instruct-2507',
  name: 'Qwen3 4B 2507',
  publisher: 'Alibaba Qwen',
  parameters: '4B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen3-4B-Instruct-2507-ONNX', revision: '2bdfea4417ee390519508a6bc7dfbc36f3572981' },
  architecture: {
    family: 'qwen3',
    parameterCount: 4_022_468_096,
    layers: 36,
    hiddenSize: 2560,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 262144,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 9117040, sha256: 'e7a95fce95bf5b0946d0ddb3f9d7caa030b7e850bbe92b0edb26bcf563e9f3d5' },
  promptFormat: { kind: 'chatml', stopTokens: CHATML_STOPS },
  generation: QWEN3_06B.generation,
  quality: {
    rank: 9,
    summary: 'The July 2025 update of Qwen3 4B: much stronger at reasoning, knowledge and instructions; answers directly.',
    scores: [
      { benchmark: 'MMLU-Pro', score: 69.6, source: QWEN3_2507_CARD },
      { benchmark: 'IFEval', score: 83.4, source: QWEN3_2507_CARD },
      { benchmark: 'GPQA', score: 62.0, source: QWEN3_2507_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 437655, sha256: '6e603b4d2324fa94a1c1fc927d1f985da78413afcffa27b02b008d909eccc4a8' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2094347264, sha256: 'd2b17463255a120a6d4e88c60e24bb6be6607ab6e34dd89b62234879bfd6ce0d' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 794787840, sha256: '6c0d3b7ee94fa08c2dc766dfa4c5b790e492077701847b40fa001f648f3ebded' }],
      kvCacheType: 'float16',
      largestTensorBytes: 777912320,
      computeOps: 542,
    },
  ],
};

export const GEMMA3_4B_IT: ModelSpec = {
  id: 'gemma-3-4b-it',
  name: 'Gemma 3 4B',
  publisher: 'Google',
  parameters: '4B',
  license: 'Gemma Terms of Use',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/gemma-3-4b-it-ONNX', revision: '944e625cdc227b1f1c41db1ae6bba655cda90104' },
  architecture: {
    family: 'gemma3',
    parameterCount: 3_880_000_000,
    layers: 34,
    hiddenSize: 2560,
    kvHeads: 4,
    headDim: 256,
    vocabSize: 262208,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 20323105, sha256: 'b472c7c2d1a2ac607daa53c794575e59504c06b39579a0f2399bb95cea6a4949' },
  promptFormat: GEMMA_FORMAT,
  generation: GEMMA_GENERATION,
  quality: {
    rank: 8,
    summary: 'Google’s 4B model: natural, well-written replies and strong instruction following; many languages.',
    scores: [
      { benchmark: 'IFEval', score: 90.2, source: GEMMA3_REPORT },
      { benchmark: 'GSM8K', score: 89.2, source: GEMMA3_REPORT },
      { benchmark: 'MMLU-Pro', score: 43.6, source: GEMMA3_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/decoder_model_merged_q4.onnx', sizeBytes: 467689, sha256: '70269ef03c0cdb9c1f247c4ac76e75036ee7df09757b9ac165308fd80738df71' },
      externalData: [{ path: 'onnx/decoder_model_merged_q4.onnx_data', sizeBytes: 2093692928, sha256: '4d339ace22e320b447a52e7861367328efd4a4b742920ed0bab0c0631e99b46c' }, { path: 'onnx/decoder_model_merged_q4.onnx_data_1', sizeBytes: 661772800, sha256: 'c854198bff7eeac9948f675148f3b4f0582e433f9bc1326a3d3af55cf82cbb02' }],
      embeddings: { graph: { path: 'onnx/embed_tokens_q4.onnx', sizeBytes: 858, sha256: '152862f7d988a855514daa3e88643cd15288f88a395930e69bf765a5679e8ccf' }, externalData: [{ path: 'onnx/embed_tokens_q4.onnx_data', sizeBytes: 430021120, sha256: '6b137a3b8184478ffd73c2e9e9af38874f9dfd8a16401cc095093dc10fef5896' }] },
      kvCacheType: 'float32',
      largestTensorBytes: 335626240,
      computeOps: 617,
    },
  ],
};

export const SMOLLM2_135M_INSTRUCT: ModelSpec = {
  id: 'smollm2-135m-instruct',
  name: 'SmolLM2 135M',
  publisher: 'Hugging Face',
  parameters: '135M',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'HuggingFaceTB/SmolLM2-135M-Instruct', revision: '12fd25f77366fa6b3b4b768ec3050bf629380bac' },
  architecture: {
    family: 'llama',
    parameterCount: 134_515_008,
    layers: 30,
    hiddenSize: 576,
    kvHeads: 3,
    headDim: 64,
    vocabSize: 49152,
    maxContextTokens: 8192,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 2104556, sha256: '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c' },
  promptFormat: { kind: 'chatml', defaultSystemPrompt: SMOLLM2_360M_INSTRUCT.promptFormat.defaultSystemPrompt, stopTokens: ['<|im_end|>'] },
  generation: SMOLLM2_360M_INSTRUCT.generation,
  quality: {
    rank: 0,
    summary: 'The smallest model here: quick to download and fast anywhere, but only for very simple requests.',
    scores: [
      { benchmark: 'IFEval (average)', score: 29.9, source: SMOLLM2_CARD },
      { benchmark: 'MMLU (cloze)', score: 29.3, source: SMOLLM2_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 117691126, sha256: '9358cd4ce037c304621f8c194a525607ae7c5ea73239fcae4c21bd02f2e34ff7' },
      kvCacheType: 'float16',
      largestTensorBytes: 56623104,
      computeOps: 452,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 182068553, sha256: '933577110303a2964096d19b6f15d3b4639bef7f99481ac0b61d9f3ad72f392a' },
      kvCacheType: 'float32',
      largestTensorBytes: 113246208,
      computeOps: 452,
    },
  ],
};

export const GEMMA3_270M_IT: ModelSpec = {
  id: 'gemma-3-270m-it',
  name: 'Gemma 3 270M',
  publisher: 'Google',
  parameters: '270M',
  license: 'Gemma Terms of Use',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/gemma-3-270m-it-ONNX', revision: '2dbbfdb1b59bd034eb959428c6a7da9dd7ea27f0' },
  architecture: {
    family: 'gemma3',
    parameterCount: 268_098_176,
    layers: 18,
    hiddenSize: 640,
    kvHeads: 1,
    headDim: 256,
    vocabSize: 262144,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 20323013, sha256: '55da1312bdf1d7d8fe8d9d1b3eed04086261149e6034e0ac3f8c633b67f5aac8' },
  promptFormat: GEMMA_FORMAT,
  generation: GEMMA_GENERATION,
  quality: {
    rank: 0,
    summary: 'Google’s tiny model, made to be fine-tuned for one job; as a general assistant it handles only simple requests.',
    scores: [
      { benchmark: 'IFEval', score: 51.2, source: GEMMA3_270M_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 330717, sha256: '01d620bf3c2c1d5b843a5c92a4456ce1f4c63f6ceb814da26a3420e8aad2b4a4' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 272626176, sha256: '23088037c5972441c6a6033f5fdcca41c96bfea9b0bf7e1abd6038ba42e45287' }],
      kvCacheType: 'float16',
      largestTensorBytes: 83886080,
      computeOps: 365,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 242522, sha256: 'cb3615746a988839308507933d4e0501ff695e747a0a0ce15c0ef53083f8ec54' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 322933248, sha256: '25c4a14af32fc1011a90504076024b0e1f65f915e9690b44a5484200d2e20851' }],
      kvCacheType: 'float32',
      largestTensorBytes: 83886080,
      computeOps: 329,
    },
  ],
};

export const LFM25_350M: ModelSpec = {
  id: 'lfm2.5-350m',
  name: 'LFM2.5 350M',
  publisher: 'Liquid AI',
  parameters: '350M',
  license: 'LFM Open License v1.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'LiquidAI/LFM2.5-350M-ONNX', revision: 'd11593fd9eb408e322667926656598896c2d5ff9' },
  architecture: {
    family: 'lfm2',
    parameterCount: 354_483_968,
    layers: 16,
    hiddenSize: 1024,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxContextTokens: 128000,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 3297793, sha256: '29d43b4be8e8a896fefd7cd836ca6d6b4eedd249f823866ce0453b368e646f49' },
  promptFormat: LFM_FORMAT,
  generation: LFM_GENERATION,
  quality: {
    rank: 1,
    summary: 'Liquid AI’s small hybrid model: follows instructions well for its size, with little conversation memory.',
    scores: [
      { benchmark: 'IFEval', score: 76.96, source: LFM25_350M_CARD },
      { benchmark: 'MMLU-Pro', score: 20.01, source: LFM25_350M_CARD },
      { benchmark: 'GPQA Diamond', score: 30.64, source: LFM25_350M_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 182827, sha256: '3012c28a119828561c90196331435d91c43c1a6ab4898ac79715585fff1dff85' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 254965760, sha256: '9256ecd417b801b441d926ebe5ead4ebf50ac1a5a0f8f7914cf60d9d6452ef69' }],
      kvCacheType: 'float16',
      largestTensorBytes: 33554432,
      computeOps: 268,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 183442, sha256: 'd1a705712e93aafaba1346b32245fa59e7857a46e2272003c0e8c524977e0de8' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 293629952, sha256: '71ec6ad38a4c463dcb3dba671d06a1d9861be3a23e51290d818b95c0b7d2a5db' }],
      kvCacheType: 'float32',
      largestTensorBytes: 33554432,
      computeOps: 268,
    },
  ],
};

export const GRANITE4_350M: ModelSpec = {
  id: 'granite-4.0-350m',
  name: 'Granite 4.0 350M',
  publisher: 'IBM',
  parameters: '350M',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/granite-4.0-350m-ONNX-web', revision: '62fc8f345c8aed0e1a28f856dc7e79b4cdb55cae' },
  architecture: {
    family: 'granite',
    parameterCount: 352_000_000,
    layers: 28,
    hiddenSize: 1024,
    kvHeads: 4,
    headDim: 64,
    vocabSize: 100352,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 4131199, sha256: 'ea076a8bccab905484daa97acc7021a915fe30c1ad48fa85bd4b82a5d1ceb797' },
  promptFormat: GRANITE_FORMAT,
  generation: LLAMA32_1B_INSTRUCT.generation,
  quality: {
    rank: 1,
    summary: 'IBM’s smallest Granite: a compact business-style assistant for simple tasks.',
    scores: [
      { benchmark: 'IFEval (average)', score: 55.4, source: GRANITE4_CARD },
      { benchmark: 'MMLU (5-shot)', score: 35.01, source: GRANITE4_CARD },
      { benchmark: 'GSM8K (8-shot)', score: 30.71, source: GRANITE4_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 272846, sha256: '17e171e1d30f1c644eda76d673e662996ff6b89cdb22a11e2ecadde9c158535f' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 575639552, sha256: 'c322cb280cd7168e37d5d187656973ac7ccecc1465c29f52c5547b2f6e557799' }],
      kvCacheType: 'float32',
      largestTensorBytes: 411041792,
      computeOps: 424,
    },
  ],
};

export const QWEN25_CODER_05B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-coder-0.5b-instruct',
  name: 'Qwen2.5 Coder 0.5B',
  publisher: 'Alibaba Qwen',
  parameters: '0.5B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen2.5-Coder-0.5B-Instruct', revision: '604bb1988ff789d5024d589c63cf9ae81c0c8ab5' },
  architecture: {
    family: 'qwen2',
    parameterCount: 494_032_768,
    layers: 24,
    hiddenSize: 896,
    kvHeads: 2,
    headDim: 64,
    vocabSize: 151936,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 11421896, sha256: '9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa' },
  promptFormat: { kind: 'chatml', defaultSystemPrompt: QWEN25_05B_INSTRUCT.promptFormat.defaultSystemPrompt, stopTokens: CHATML_STOPS },
  generation: QWEN25_05B_INSTRUCT.generation,
  quality: {
    rank: 1,
    summary: 'For code, on small devices: short functions and explanations. Weak at everything else.',
    scores: [
      { benchmark: 'HumanEval', score: 61.6, source: QWEN25_CODER_REPORT },
      { benchmark: 'MBPP', score: 52.4, source: QWEN25_CODER_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 554935833, sha256: '60c076ac0d3910881fe0cad75997e803459a5f71bb74bcbcd82407ef83db7ba0' },
      kvCacheType: 'float16',
      largestTensorBytes: 272269312,
      computeOps: 438,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 862087110, sha256: '2df7ea9739c3c26bd9d2cad0815768a5a0a985d4b5548b7de80c5e85e9a97b47' },
      kvCacheType: 'float32',
      largestTensorBytes: 544538624,
      computeOps: 290,
    },
  ],
};

export const LFM2_700M: ModelSpec = {
  id: 'lfm2-700m',
  name: 'LFM2 700M',
  publisher: 'Liquid AI',
  parameters: '0.7B',
  license: 'LFM Open License v1.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/LFM2-700M-ONNX', revision: 'bdfeecd9300145cc1696702b74234d73f683622f' },
  architecture: {
    family: 'lfm2',
    parameterCount: 742_489_344,
    layers: 16,
    hiddenSize: 1536,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxContextTokens: 128000,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 3296940, sha256: 'bec04fe03d27dbcb6276b46e7d2325433323a8132a2f0404ea6bbf98250a9211' },
  promptFormat: LFM_FORMAT,
  generation: LFM_GENERATION,
  quality: {
    rank: 2,
    summary: 'Liquid AI’s hybrid model: good at following instructions, quick, with little conversation memory.',
    scores: [
      { benchmark: 'IFEval', score: 72.23, source: LFM2_CARD },
      { benchmark: 'MMLU', score: 49.9, source: LFM2_CARD },
      { benchmark: 'GSM8K', score: 46.4, source: LFM2_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 183018, sha256: '95ce2f28ec97b145913e7b474209c36bf5a174916fe6a13fd205531db7e24bcf' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 496104448, sha256: 'b0cd094b1731ce79e5bcc140f874147b1dc4554248ebcf6ed27c0126ad99776a' }],
      kvCacheType: 'float16',
      largestTensorBytes: 50331648,
      computeOps: 268,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 183625, sha256: '496549029747546240c69a5d11221ef3edce8c6d6f41051c42244ba47aedcefb' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 559081472, sha256: 'c063f8b36e0aa87cf659cd8c15ac23178fc90a2390f708d556d1d12d59cd2715' }],
      kvCacheType: 'float32',
      largestTensorBytes: 50331648,
      computeOps: 268,
    },
  ],
};

export const GEMMA3_1B_IT: ModelSpec = {
  id: 'gemma-3-1b-it',
  name: 'Gemma 3 1B',
  publisher: 'Google',
  parameters: '1B',
  license: 'Gemma Terms of Use',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/gemma-3-1b-it-ONNX', revision: 'a58439f40017d3b99c7d378ff525e54e0ba08ebf' },
  architecture: {
    family: 'gemma3',
    parameterCount: 999_885_952,
    layers: 26,
    hiddenSize: 1152,
    kvHeads: 1,
    headDim: 256,
    vocabSize: 262144,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 20323013, sha256: '55da1312bdf1d7d8fe8d9d1b3eed04086261149e6034e0ac3f8c633b67f5aac8' },
  promptFormat: GEMMA_FORMAT,
  generation: GEMMA_GENERATION,
  quality: {
    rank: 3,
    summary: 'Google’s small model: natural, well-written replies and good instruction following; weaker on facts.',
    scores: [
      { benchmark: 'IFEval', score: 80.2, source: GEMMA3_REPORT },
      { benchmark: 'GSM8K', score: 62.8, source: GEMMA3_REPORT },
      { benchmark: 'MMLU-Pro', score: 14.7, source: GEMMA3_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 461341, sha256: '522d1f0e1d1fd0065ca02d61a0bf17a43701c1835b4ebf0c4628b73f1a6fb796' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 763067904, sha256: '6fb9bc8dd4b93a040bcea60c953c2e24f60f9d52e708befec077f6ab60baa007' }],
      kvCacheType: 'float16',
      largestTensorBytes: 150994944,
      computeOps: 473,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 347363, sha256: '69686023e5892376e38fcbcdd0c77af432c55b3bcd03aee6d561bd1f04507da0' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 859106816, sha256: 'c2370070be257a98d50e17d81be13e18304c39e7e6d9d1416f8f883681d2a17b' }],
      kvCacheType: 'float32',
      largestTensorBytes: 150994944,
      computeOps: 473,
    },
  ],
};

export const GRANITE4_1B: ModelSpec = {
  id: 'granite-4.0-1b',
  name: 'Granite 4.0 1B',
  publisher: 'IBM',
  parameters: '1.6B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/granite-4.0-1b-ONNX-web', revision: 'd65b1a480bfab70bc828208d7903cf9de4bdfaa1' },
  architecture: {
    family: 'granite',
    parameterCount: 1_630_000_000,
    layers: 40,
    hiddenSize: 2048,
    kvHeads: 4,
    headDim: 128,
    vocabSize: 100352,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 4131199, sha256: 'ea076a8bccab905484daa97acc7021a915fe30c1ad48fa85bd4b82a5d1ceb797' },
  promptFormat: GRANITE_FORMAT,
  generation: LLAMA32_1B_INSTRUCT.generation,
  quality: {
    rank: 4,
    summary: 'IBM’s small Granite: dependable instruction following and maths for its size.',
    scores: [
      { benchmark: 'IFEval (average)', score: 77.38, source: GRANITE4_CARD },
      { benchmark: 'MMLU (5-shot)', score: 59.39, source: GRANITE4_CARD },
      { benchmark: 'GSM8K (8-shot)', score: 76.35, source: GRANITE4_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 389658, sha256: '4e125c4855f430c70581f457d6d5d21803dca366b628cf8931b9708c480e0523' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 1781145600, sha256: '3d32a0c75d82304bacf353b2c531e2dbb958e51286cfb426123e6d4918a9a92c' }],
      kvCacheType: 'float32',
      largestTensorBytes: 822083584,
      computeOps: 604,
    },
  ],
};

export const LFM25_12B_INSTRUCT: ModelSpec = {
  id: 'lfm2.5-1.2b-instruct',
  name: 'LFM2.5 1.2B',
  publisher: 'Liquid AI',
  parameters: '1.2B',
  license: 'LFM Open License v1.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'LiquidAI/LFM2.5-1.2B-Instruct-ONNX', revision: '10f72e70abf67ac0fd7ebf15bc5854726891d864' },
  architecture: {
    family: 'lfm2',
    parameterCount: 1_170_340_608,
    layers: 16,
    hiddenSize: 2048,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxContextTokens: 128000,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 3297799, sha256: '0a1f2cb9fc2769030ca1f4207e81c1af493319373d41b8d3cf5be53860c13760' },
  promptFormat: LFM_FORMAT,
  generation: LFM_GENERATION,
  quality: {
    rank: 4,
    summary: 'Liquid AI’s newest small model: very good at following instructions for its size, and quick.',
    scores: [
      { benchmark: 'IFEval', score: 86.23, source: LFM25_12B_CARD },
      { benchmark: 'MMLU-Pro', score: 44.35, source: LFM25_12B_CARD },
      { benchmark: 'GPQA', score: 38.89, source: LFM25_12B_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 182795, sha256: 'a9986ad188200507342ac32f727aa4691edb5428b0aa8c4a9fbdf0c85a6fe667' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 760279040, sha256: '46cfacc12941150620a3f644a5269e9baebd75d681cfa09cadeef71b8ed64ac2' }],
      kvCacheType: 'float16',
      largestTensorBytes: 67108864,
      computeOps: 265,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 183173, sha256: 'c760a2d8be1d19c71c470ef294c1220ef85b5b3a0e31f12a363655ea7fab1d58' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 850059264, sha256: 'd9666c44e2acc32f06c9351f9e7c4fd66bc060d88ca8e7f954e836c6845f7488' }],
      kvCacheType: 'float32',
      largestTensorBytes: 67108864,
      computeOps: 265,
    },
  ],
};

export const QWEN25_MATH_15B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-math-1.5b-instruct',
  name: 'Qwen2.5 Math 1.5B',
  publisher: 'Alibaba Qwen',
  parameters: '1.5B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen2.5-Math-1.5B-Instruct', revision: '29b745e87b28faa666ea598ee43b4db439144be6' },
  architecture: {
    family: 'qwen2',
    parameterCount: 1_543_714_304,
    layers: 28,
    hiddenSize: 1536,
    kvHeads: 2,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 4096,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 7031673, sha256: 'a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf' },
  promptFormat: { kind: 'chatml', defaultSystemPrompt: 'Please reason step by step, and put your final answer within \\boxed{}.', stopTokens: CHATML_STOPS },
  generation: QWEN25_05B_INSTRUCT.generation,
  quality: {
    rank: 3,
    summary: 'For maths problems in English and Chinese, worked step by step. Not for general chat.',
    scores: [
      { benchmark: 'MATH (CoT)', score: 75.8, source: QWEN25_MATH_REPORT },
      { benchmark: 'GSM8K (CoT)', score: 84.8, source: QWEN25_MATH_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 1758206404, sha256: 'c00fb4d8c11c5888845408916d126d2c636c2c4f0f4b96ed4bb59531a79c5594' },
      kvCacheType: 'float32',
      largestTensorBytes: 933494784,
      computeOps: 1248,
    },
  ],
};

export const LFM2_26B: ModelSpec = {
  id: 'lfm2-2.6b',
  name: 'LFM2 2.6B',
  publisher: 'Liquid AI',
  parameters: '2.6B',
  license: 'LFM Open License v1.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/LFM2-2.6B-ONNX', revision: 'cd7301a675f9b6eab8c789488d9b22333b2fd023' },
  architecture: {
    family: 'lfm2',
    parameterCount: 2_569_272_320,
    layers: 30,
    hiddenSize: 2048,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxContextTokens: 128000,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 3296920, sha256: '0d97ddf099eae6c0b4392bdffa1256762eb8621de54e04429e0719965036c3d7' },
  promptFormat: LFM_FORMAT,
  generation: LFM_GENERATION,
  quality: {
    rank: 6,
    summary: 'Liquid AI’s larger hybrid model: good instruction following and maths, faster than most models its size.',
    scores: [
      { benchmark: 'IFEval', score: 79.56, source: LFM2_26B_CARD },
      { benchmark: 'MMLU', score: 64.42, source: LFM2_26B_CARD },
      { benchmark: 'GSM8K', score: 82.41, source: LFM2_26B_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 322750, sha256: 'd98c4f6cf16c142f928127c01be89b0199d7a0c8223c60a756bfde21afe33878' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 1654910976, sha256: '3dd89d13c6c716bba3fe196f9b6cd22abc51a1a476efb70a8c0151b14a5f8cfe' }],
      kvCacheType: 'float16',
      largestTensorBytes: 268435456,
      computeOps: 490,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 323506, sha256: '26cecca7cc85cad6d58766a1081d899e7cd1388af7c59afdd5c0d74da3d20429' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 2092425216, sha256: '5f80e6c5eb5787c0c5825b8772f614113b09e6ac23be4dba58c8447c963bf157' }],
      kvCacheType: 'float32',
      largestTensorBytes: 536870912,
      computeOps: 490,
    },
  ],
};

export const QWEN25_CODER_3B_INSTRUCT: ModelSpec = {
  id: 'qwen2.5-coder-3b-instruct',
  name: 'Qwen2.5 Coder 3B',
  publisher: 'Alibaba Qwen',
  parameters: '3B',
  license: 'Qwen Research License',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Qwen2.5-Coder-3B-Instruct', revision: '0e076a9c5291f8c74b3ce1558f21336b18f94739' },
  architecture: {
    family: 'qwen2',
    parameterCount: 3_085_938_688,
    layers: 36,
    hiddenSize: 2048,
    kvHeads: 2,
    headDim: 128,
    vocabSize: 151936,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 11421896, sha256: '9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa' },
  promptFormat: { kind: 'chatml', defaultSystemPrompt: QWEN25_05B_INSTRUCT.promptFormat.defaultSystemPrompt, stopTokens: CHATML_STOPS },
  generation: QWEN25_05B_INSTRUCT.generation,
  quality: {
    rank: 5,
    summary: 'For code: writes, fixes and explains code well. Weaker than general models at everything else.',
    scores: [
      { benchmark: 'HumanEval', score: 84.1, source: QWEN25_CODER_REPORT },
      { benchmark: 'MBPP', score: 73.6, source: QWEN25_CODER_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 271268174, sha256: '003639a5dd89da951693ac9ec29c16ff1c557d6669bf830cca9eb4546cce100f' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2096365568, sha256: '2216b89a17ab28efcf5700446a870464ee418fec2f5388198bb7a107c2cf0288' }],
      kvCacheType: 'float16',
      largestTensorBytes: 622329856,
      computeOps: 654,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 1098040858, sha256: 'cf4854a472f6a7b800a1a9967d8a6faa5dd451fb4b012ac8c5712daf2959610e' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 2093154304, sha256: 'cd6ce254e9d59e14e015fbb9725d018c8d8bda3588abe72c31f4bb5e2aefa68b' }],
      kvCacheType: 'float32',
      largestTensorBytes: 1244659712,
      computeOps: 434,
    },
  ],
};

export const GRANITE4_MICRO: ModelSpec = {
  id: 'granite-4.0-micro',
  name: 'Granite 4.0 Micro',
  publisher: 'IBM',
  parameters: '3B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/granite-4.0-micro-ONNX-web', revision: '055c9bd9db8f5e47bebd17246b0c614ffe5e8a13' },
  architecture: {
    family: 'granite',
    parameterCount: 3_400_000_000,
    layers: 40,
    hiddenSize: 2560,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 100352,
    maxContextTokens: 131072,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 4131199, sha256: 'ea076a8bccab905484daa97acc7021a915fe30c1ad48fa85bd4b82a5d1ceb797' },
  promptFormat: GRANITE_MICRO_FORMAT,
  generation: LLAMA32_1B_INSTRUCT.generation,
  quality: {
    rank: 7,
    summary: 'IBM’s 3B Granite: strong instruction following and maths; a 2.3 GB download.',
    scores: [
      { benchmark: 'IFEval (average)', score: 82.31, source: GRANITE4_CARD },
      { benchmark: 'MMLU (5-shot)', score: 65.98, source: GRANITE4_CARD },
      { benchmark: 'GSM8K (8-shot)', score: 85.45, source: GRANITE4_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 391777, sha256: 'c261780a57c22ff7b9107facea0c035a3be0501248e81da6da96a35a13ad62c0' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2088129536, sha256: '847d3496fd4d6a2cf70c5dd5920fa50b47754f0c963c8cec92563508801d1eb7' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 212336640, sha256: '4863091011d2cceb4325f5d0e7eac90c28b602566951cce8742ce67ee8442d98' }],
      kvCacheType: 'float16',
      largestTensorBytes: 513802240,
      computeOps: 604,
    },
  ],
};

export const PHI3_MINI_4K_INSTRUCT: ModelSpec = {
  id: 'phi-3-mini-4k-instruct',
  name: 'Phi-3 Mini',
  publisher: 'Microsoft',
  parameters: '3.8B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'microsoft/Phi-3-mini-4k-instruct-onnx-web', revision: '80a2792f5bf861528ce9b449b3230f1bd3fdc759' },
  architecture: {
    family: 'phi3',
    parameterCount: 3_821_079_552,
    layers: 32,
    hiddenSize: 3072,
    kvHeads: 32,
    headDim: 96,
    vocabSize: 32064,
    maxContextTokens: 4096,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 1937870, sha256: '8e9e94300b0c19471520eca5e6cacc4a522c7da823edac3b2cdd1152b4fb3ae2' },
  promptFormat: PHI3_FORMAT,
  generation: PHI_GENERATION,
  quality: {
    rank: 6,
    summary: 'Microsoft’s earlier 3.8B model: good reasoning and maths, English only; 4K context at most.',
    scores: [
      { benchmark: 'MMLU (5-shot)', score: 70.9, source: PHI3_CARD },
      { benchmark: 'GSM8K (CoT)', score: 85.7, source: PHI3_CARD },
      { benchmark: 'GPQA', score: 30.6, source: PHI3_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 838080425, sha256: 'e726f1c300be9714987ebf7cb047afcb002f6ce9a28af1771d05d5d1d4eeaf48' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 1454211072, sha256: '62eb6a6870961b0039732d574779e44a7de42e26c04ca1433759728d644cc00f' }],
      kvCacheType: 'float16',
      largestTensorBytes: 197001216,
      computeOps: 488,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 1059602332, sha256: '16b8e5d28a757c37bbfa7d9420fd094c0c20e3615ca3c203b5b9501015045c8f' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 1663524864, sha256: '41d30b87f06b52e6b24c4e2e65a6a14e5c9fb5bc6f495fac17b19c6bc7875ff5' }],
      kvCacheType: 'float32',
      largestTensorBytes: 394002432,
      computeOps: 488,
    },
  ],
};

export const OLMO3_7B_INSTRUCT: ModelSpec = {
  id: 'olmo-3-7b-instruct',
  name: 'OLMo 3 7B',
  publisher: 'Ai2',
  parameters: '7B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Olmo-3-7B-Instruct-ONNX', revision: '64cf5a8d2dac83c3b889d44c422de2ae3e2b2e7d' },
  architecture: {
    family: 'olmo3',
    parameterCount: 7_298_011_136,
    layers: 32,
    hiddenSize: 4096,
    kvHeads: 32,
    headDim: 128,
    vocabSize: 100278,
    maxContextTokens: 65536,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 4119438, sha256: 'e6297cf3c369c31fc24313131019d0cf6d7ddd137ea1e75d7a65eca7edfcd522' },
  promptFormat: OLMO3_FORMAT,
  generation: OLMO3_GENERATION,
  quality: {
    rank: 9,
    summary: 'Ai2’s fully open 7B model: strong instruction following and maths. A 3.8 GB download that needs a capable GPU.',
    scores: [
      { benchmark: 'IFEval', score: 85.6, source: OLMO3_CARD },
      { benchmark: 'MMLU', score: 69.1, source: OLMO3_CARD },
      { benchmark: 'MATH', score: 87.3, source: OLMO3_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 414480, sha256: '26f2b043f4567287ac00c7669e38b1356a5feb7ae5b2ab37a5e91ed6dc6fc65f' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2077432032, sha256: '32aeeba1ada2d4777d5b5e0b3b37bb334644e53ab021f846a15ea6f5f7aaccc1' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 1732204768, sha256: '39c21ba9fd832cc978da7670a01a865f83998fced33c0b4e41b7700e7348bbb3' }],
      kvCacheType: 'float16',
      largestTensorBytes: 205369344,
      computeOps: 547,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 363850, sha256: 'b84a33e314a2a0ac7e9a0345047beb1a84f131978dcc58dc8593c5998980daee' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 2096617056, sha256: '881451b57726b3eda20f9a2f03dc2edee17287224b7311f11155e09941925971' }, { path: 'onnx/model_q4.onnx_data_1', sizeBytes: 1844344416, sha256: 'e78bc82239d56f6e91dc959539bba406e79c9cef05eccf32767b184308f2e394' }],
      kvCacheType: 'float32',
      largestTensorBytes: 205369344,
      computeOps: 547,
    },
  ],
};

export const LFM2_8B_A1B: ModelSpec = {
  id: 'lfm2-8b-a1b',
  name: 'LFM2 8B-A1B',
  publisher: 'Liquid AI',
  parameters: '8.3B (1.5B active)',
  license: 'LFM Open License v1.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/LFM2-8B-A1B-ONNX', revision: 'c9bf5d746af56d087076debc9d0b59f73434d8c8' },
  architecture: {
    family: 'lfm2',
    parameterCount: 8_340_000_000,
    activeParameterCount: 1_500_000_000,
    layers: 24,
    hiddenSize: 2048,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxContextTokens: 128000,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 3297530, sha256: '40b989ce8875ab2d4be98959e764b4dc761b8d3bfd1dcfe48f8d69f559eac3f4' },
  promptFormat: LFM_FORMAT,
  generation: LFM_GENERATION,
  quality: {
    rank: 8,
    summary: 'Liquid AI’s mixture-of-experts model: the knowledge of an 8B model at nearly the speed of a 1.5B one, but a 4.8 GB download.',
    scores: [
      { benchmark: 'IFEval', score: 77.58, source: LFM2_8B_CARD },
      { benchmark: 'MMLU', score: 64.84, source: LFM2_8B_CARD },
      { benchmark: 'GSM8K', score: 84.38, source: LFM2_8B_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 298018, sha256: 'fe3ac00b2877bfa4a18aba49971d90c41a53af250ce258636be6f164ecd9ee2d' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2076209152, sha256: 'd2aa8b7ae553d6acc3af770b4f7624dccf6cce3fb0a0c8e417afe162ccd1ad05' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 1981808640, sha256: '77de2b5d304514a38f8ffd9a08cfd1fe1fc810202e83bc55df76e9a8551ce4a1' }, { path: 'onnx/model_q4f16.onnx_data_2', sizeBytes: 726185984, sha256: '462d873930e717de9d2f106e60ca2ddb62458360139afdc528c873f2b0f3e33e' }],
      kvCacheType: 'float16',
      largestTensorBytes: 117440512,
      computeOps: 484,
    },
    {
      id: 'onnx-q4',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4',
      requiredGpuFeatures: [],
      graph: { path: 'onnx/model_q4.onnx', sizeBytes: 300329, sha256: '99b1c373c7d1c5cbe71645992f620540ffa24542bdea2b80dd6c6843454e3ab3' },
      externalData: [{ path: 'onnx/model_q4.onnx_data', sizeBytes: 2086723584, sha256: '1e5ee60e6cbaf6d70ffd000fff2667a0b2e4ffd454e0f90ae05edcc617745caf' }, { path: 'onnx/model_q4.onnx_data_1', sizeBytes: 1996488704, sha256: '5d3846396d300289e7813da521ed9fc2df9fe4012a773d678daf447705f12d10' }, { path: 'onnx/model_q4.onnx_data_2', sizeBytes: 1239029760, sha256: 'd0311083ed54febf6439bd72ee115961c4235f3e1c6b3d9ded34c08b640f6ced' }],
      kvCacheType: 'float32',
      largestTensorBytes: 117440512,
      computeOps: 484,
    },
  ],
};

export const LFM2_24B_A2B: ModelSpec = {
  id: 'lfm2-24b-a2b',
  name: 'LFM2 24B-A2B',
  publisher: 'Liquid AI',
  parameters: '24B (2.3B active)',
  license: 'LFM Open License v1.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'LiquidAI/LFM2-24B-A2B-ONNX', revision: 'e6fb57d2e0370252b077858efcdd906d1c4b08df' },
  architecture: {
    family: 'lfm2',
    parameterCount: 24_000_000_000,
    activeParameterCount: 2_300_000_000,
    layers: 40,
    hiddenSize: 2048,
    kvHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxContextTokens: 128000,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 4733389, sha256: 'df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd' },
  promptFormat: LFM_FORMAT,
  generation: LFM_GENERATION,
  quality: {
    rank: 10,
    summary: 'Liquid AI’s largest mixture-of-experts model. A 13.5 GB download: only for machines with a lot of GPU memory.',
    scores: [],
  },
  experimental: 'Untested: its weights alone are 13.5 GB, so it needs a Mac with 32 GB or more (or a GPU with 16 GB), more than any machine we test on.',
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/model_q4f16.onnx', sizeBytes: 495227, sha256: '18725e7bab28d23cf67a01718909c1d99e79701d967a3990f20241316c5e60fb' },
      externalData: [{ path: 'onnx/model_q4f16.onnx_data', sizeBytes: 2019864576, sha256: '12f9ad0a641420cf6afb9c4ddae3c0efff3ad53e25a95859bd615416a74a34e6' }, { path: 'onnx/model_q4f16.onnx_data_1', sizeBytes: 2038431744, sha256: 'ae5a54829eadaa5802d946fcda7a7961884079c3441b8c2f5ddce4367349064c' }, { path: 'onnx/model_q4f16.onnx_data_2', sizeBytes: 2038431744, sha256: '868144536237c0556720922cf223c6f91a610be33e7b97b900000422781a43ae' }, { path: 'onnx/model_q4f16.onnx_data_3', sizeBytes: 2038431744, sha256: '849719320cc78d6c6bf23c75eade5cc16079a88d86563656e1fa79d6941614b7' }, { path: 'onnx/model_q4f16.onnx_data_4', sizeBytes: 2038431744, sha256: 'aa7b9076059dbd97321ccfb74712e55264d8e4aad251392106c678b5553773d0' }, { path: 'onnx/model_q4f16.onnx_data_5', sizeBytes: 2038431744, sha256: 'b82d51164169742513933204011beb2bdbf7d7ad7e5da2a1d1984aae2d0ae4d0' }, { path: 'onnx/model_q4f16.onnx_data_6', sizeBytes: 1297969152, sha256: 'b301737d179d7142ac10bfa9113496f51d8ca9ce1effd6d5de2834c542ca3894' }],
      kvCacheType: 'float16',
      largestTensorBytes: 201326592,
      computeOps: 808,
    },
  ],
};

export const PHI4_MINI_REASONING: ModelSpec = {
  id: 'phi-4-mini-reasoning',
  name: 'Phi-4 Mini Reasoning',
  publisher: 'Microsoft',
  parameters: '3.8B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'microsoft/Phi-4-mini-reasoning-onnx', revision: '8d5c831bede5ea4d224a9c82f1b30ddee9a40dd4' },
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
  tokenizer: { path: 'gpu/gpu-int4-rtn-block-32/tokenizer.json', sizeBytes: 15524195, sha256: '166c68da2b6ef97ff617410fc0dc55f422b33be8c2c2409c8fa13ba137d84c61' },
  promptFormat: PHI4_MINI_REASONING_FORMAT,
  generation: REASONING_GENERATION,
  quality: {
    rank: 7,
    summary: 'Thinks step by step before answering: strong at maths for its size; little general knowledge.',
    scores: [
      { benchmark: 'MATH-500', score: 94.6, source: PHI4_MINI_REASONING_CARD },
      { benchmark: 'AIME 2024', score: 57.5, source: PHI4_MINI_REASONING_CARD },
      { benchmark: 'GPQA Diamond', score: 52.0, source: PHI4_MINI_REASONING_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'gpu/gpu-int4-rtn-block-32/model.onnx', sizeBytes: 287586, sha256: '59ab70c351e50301c1cf4cca3f23176bba73e43f0d94cea26d471ba43b898f65' },
      externalData: [{ path: 'gpu/gpu-int4-rtn-block-32/model.onnx.data', sizeBytes: 3413194752, sha256: '5fc3832908b14538d2d53597507a10160e1307c45e153fa5822d82e6248471c3' }],
      kvCacheType: 'float16',
      largestTensorBytes: 1229193216,
      computeOps: 486,
    },
  ],
};

export const MISTRAL_7B_INSTRUCT_V02: ModelSpec = {
  id: 'mistral-7b-instruct-v0.2',
  name: 'Mistral 7B v0.2',
  publisher: 'Mistral AI',
  parameters: '7.2B',
  license: 'Apache-2.0',
  task: 'chat',
  source: { host: 'huggingface', repo: 'microsoft/mistral-7b-instruct-v0.2-ONNX', revision: '6d524105ea40470ca1897c97cc467f1f96a8a077' },
  architecture: {
    family: 'llama',
    parameterCount: 7_242_000_000,
    layers: 32,
    hiddenSize: 4096,
    kvHeads: 8,
    headDim: 128,
    vocabSize: 32000,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'onnx/directml/mistralai_Mistral-7B-Instruct-v0.2/tokenizer.json', sizeBytes: 1795303, sha256: 'fc4f0bd70b3709312d9d1d9e5ba674794b6bc5abc17429897a540f93882f25fc' },
  promptFormat: MISTRAL_FORMAT,
  generation: MISTRAL_GENERATION,
  quality: {
    rank: 7,
    summary: 'Mistral’s classic 7B chat model: fluent, general-purpose replies; older and weaker at maths than newer models.',
    scores: [],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'onnx/directml/mistralai_Mistral-7B-Instruct-v0.2/model.onnx', sizeBytes: 11999645, sha256: '298f8cc3cf1b3dfeddcc3912a1900f235499fb3e4354389cbbf08fac99666e55' },
      externalData: [{ path: 'onnx/directml/mistralai_Mistral-7B-Instruct-v0.2/model.onnx.data', sizeBytes: 3956748288, sha256: 'ca1244eb774db3cc813d308bb56e6671387c33380fa9145a61fdf40cc62df891' }],
      kvCacheType: 'float16',
      largestTensorBytes: 262144000,
      computeOps: 610,
    },
  ],
};

export const LLAMA31_8B_INSTRUCT: ModelSpec = {
  id: 'llama-3.1-8b-instruct',
  name: 'Llama 3.1 8B',
  publisher: 'Meta',
  parameters: '8B',
  license: 'Llama 3.1 Community License',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/Meta-Llama-3.1-8B-Instruct-ONNX-DirectML-GenAI-INT4', revision: 'aaaabf87557f7aa68b3100638fe69a9c9cc62290' },
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
  tokenizer: { path: 'tokenizer.json', sizeBytes: 17209920, sha256: '6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b' },
  promptFormat: LLAMA3_FORMAT,
  generation: LLAMA32_1B_INSTRUCT.generation,
  quality: {
    rank: 9,
    summary: 'Meta’s well-rounded 8B model: very good at following instructions, broad knowledge. A 5.3 GB download.',
    scores: [
      { benchmark: 'IFEval', score: 80.4, source: LLAMA31_CARD },
      { benchmark: 'MMLU (0-shot, CoT)', score: 73.0, source: LLAMA31_CARD },
      { benchmark: 'GSM8K (8-shot, CoT)', score: 84.5, source: LLAMA31_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'model.onnx', sizeBytes: 284037, sha256: 'a17089426f1d609ca4c53b50358b3f87eb37626e342500b7a5d746c97e8255a7' },
      externalData: [{ path: 'model.onnx.data', sizeBytes: 5306130432, sha256: 'bf16e53214a520b52d4e5d43b64986fccfd6cc6b90cd98447e0b81980dfccc03' }],
      kvCacheType: 'float16',
      largestTensorBytes: 1050673152,
      computeOps: 482,
    },
  ],
};

export const DEEPSEEK_R1_LLAMA_8B: ModelSpec = {
  id: 'deepseek-r1-distill-llama-8b',
  name: 'DeepSeek-R1 Llama 8B',
  publisher: 'DeepSeek',
  parameters: '8B',
  license: 'MIT (Llama 3.1 base)',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/DeepSeek-R1-Distill-Llama-8B-ONNX-DirectML-GenAI-INT4', revision: 'eb08877000b8ba75a7097c56acdf58cc49ab077d' },
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
  tokenizer: { path: 'tokenizer.json', sizeBytes: 17209530, sha256: 'd91915040cfac999d8c55f4b5bc6e67367c065e3a7a4e4b9438ce1f256addd86' },
  promptFormat: DEEPSEEK_FORMAT,
  generation: REASONING_GENERATION,
  quality: {
    rank: 9,
    summary: 'Thinks at length before answering: strong at maths and logic, slow to reply. A 5.3 GB download.',
    scores: [
      { benchmark: 'MATH-500', score: 89.1, source: DEEPSEEK_R1_REPORT },
      { benchmark: 'AIME 2024', score: 50.4, source: DEEPSEEK_R1_REPORT },
      { benchmark: 'GPQA Diamond', score: 49.0, source: DEEPSEEK_R1_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'model.onnx', sizeBytes: 284037, sha256: 'a17089426f1d609ca4c53b50358b3f87eb37626e342500b7a5d746c97e8255a7' },
      externalData: [{ path: 'model.onnx.data', sizeBytes: 5306130432, sha256: '2e8fcf924a0ae90140c0099f6c55d5aa8a8259810185afc2e2c33b0a527c488f' }],
      kvCacheType: 'float16',
      largestTensorBytes: 1050673152,
      computeOps: 482,
    },
  ],
};

export const GEMMA2_9B_IT: ModelSpec = {
  id: 'gemma-2-9b-it',
  name: 'Gemma 2 9B',
  publisher: 'Google',
  parameters: '9B',
  license: 'Gemma Terms of Use',
  task: 'chat',
  source: { host: 'huggingface', repo: 'onnx-community/gemma-2-9b-it-ONNX-DirectML-GenAI-INT4', revision: 'f0105d176397fc093f101fb73b937d2e90571d9d' },
  architecture: {
    family: 'gemma2',
    parameterCount: 9_242_000_000,
    layers: 42,
    hiddenSize: 3584,
    kvHeads: 8,
    headDim: 256,
    vocabSize: 256000,
    maxContextTokens: 8192,
  },
  tokenizer: { path: 'tokenizer.json', sizeBytes: 34362873, sha256: '5f7eee611703c5ce5d1eee32d9cdcfe465647b8aff0c1dfb3bed7ad7dbb05060' },
  promptFormat: GEMMA_FORMAT,
  generation: GEMMA_GENERATION,
  quality: {
    rank: 9,
    summary: 'Google’s 9B model: natural, well-written replies. A 7 GB download; needs a capable GPU.',
    scores: [
      { benchmark: 'MMLU', score: 72.3, source: GEMMA2_REPORT },
      { benchmark: 'Chatbot Arena Elo', score: 1187, source: GEMMA2_REPORT },
      { benchmark: 'GSM8K (pre-trained model)', score: 68.6, source: GEMMA2_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'model.onnx', sizeBytes: 411237, sha256: '7ef9b72f5b8d2333a60e42a229b02879062171776ba4e135b37ce6dd394f9114' },
      externalData: [{ path: 'model.onnx.data', sizeBytes: 7038532608, sha256: '8fde2299ee288fc1a49ed9a45e28aea844dd590d2ae2d5f8580d0bd0b2c93813' }],
      kvCacheType: 'float16',
      largestTensorBytes: 1835008000,
      computeOps: 678,
    },
  ],
};

export const PHI3_MEDIUM_4K_INSTRUCT: ModelSpec = {
  id: 'phi-3-medium-4k-instruct',
  name: 'Phi-3 Medium',
  publisher: 'Microsoft',
  parameters: '14B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'microsoft/Phi-3-medium-4k-instruct-onnx-directml', revision: '6e74e8bd07ed934dc7521e1e9b4c1a7121284193' },
  architecture: {
    family: 'phi3',
    parameterCount: 13_960_000_000,
    layers: 40,
    hiddenSize: 5120,
    kvHeads: 10,
    headDim: 128,
    vocabSize: 32064,
    maxContextTokens: 4096,
  },
  tokenizer: { path: 'directml-int4-awq-block-128/tokenizer.json', sizeBytes: 1844436, sha256: 'd0f067e1e15cd0a36ebef3668024882cb67a80b86fb4b7b4b128481f0d474db7' },
  promptFormat: PHI3_MEDIUM_FORMAT,
  generation: PHI_GENERATION,
  quality: {
    rank: 10,
    summary: 'Microsoft’s 14B model: strong reasoning and maths, English only; 4K context. A 7.5 GB download for big GPUs.',
    scores: [
      { benchmark: 'MMLU (5-shot)', score: 78.0, source: PHI3_MEDIUM_CARD },
      { benchmark: 'GSM8K (CoT)', score: 91.0, source: PHI3_MEDIUM_CARD },
      { benchmark: 'HumanEval', score: 62.2, source: PHI3_MEDIUM_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'directml-int4-awq-block-128/model.onnx', sizeBytes: 3750885, sha256: 'ec36b45ae6c94ff49935535b6c78e591e15d6fb5bf9d3e1e9e1f03da03a32285' },
      externalData: [{ path: 'directml-int4-awq-block-128/model.onnx.data', sizeBytes: 7496439040, sha256: 'd5bdb5e59d5bde62f6ac2bf0c8969d61d80ae6bffc216b171c4105b6271c382d' }],
      kvCacheType: 'float16',
      largestTensorBytes: 328335360,
      computeOps: 722,
    },
  ],
};

export const PHI4: ModelSpec = {
  id: 'phi-4',
  name: 'Phi-4',
  publisher: 'Microsoft',
  parameters: '14B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'microsoft/phi-4-onnx', revision: 'c4e245b57793e8481e0fa9252b3a45dee44a921b' },
  architecture: {
    family: 'phi3',
    parameterCount: 14_660_000_000,
    layers: 40,
    hiddenSize: 5120,
    kvHeads: 10,
    headDim: 128,
    vocabSize: 100352,
    maxContextTokens: 16384,
  },
  tokenizer: { path: 'gpu/gpu-int4-rtn-block-32/tokenizer.json', sizeBytes: 7153083, sha256: 'c612e57b2763204798271c0cdc14a373a565e6a371f1bfbd44e385a8391c9884' },
  promptFormat: PHI4_FORMAT,
  generation: PHI_GENERATION,
  quality: {
    rank: 11,
    summary: 'Microsoft’s 14B model: excellent maths, reasoning and code for its size. A 9 GB download for big GPUs.',
    scores: [
      { benchmark: 'MMLU', score: 84.8, source: PHI4_REPORT },
      { benchmark: 'MATH', score: 80.4, source: PHI4_REPORT },
      { benchmark: 'GPQA', score: 56.1, source: PHI4_REPORT },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'gpu/gpu-int4-rtn-block-32/model.onnx', sizeBytes: 358825, sha256: '94ce65a52a4ef7fcea28bb5e5a677bf57968116f68c50a5b6ba63e4439fe85c1' },
      externalData: [{ path: 'gpu/gpu-int4-rtn-block-32/model.onnx.data', sizeBytes: 8989353984, sha256: 'af8d99d133c18ac59cc84395cb8ee2a82e753c4662366a40c69ea34ff561fa19' }],
      kvCacheType: 'float16',
      largestTensorBytes: 1027604480,
      computeOps: 602,
    },
  ],
};

export const PHI4_REASONING: ModelSpec = {
  id: 'phi-4-reasoning',
  name: 'Phi-4 Reasoning',
  publisher: 'Microsoft',
  parameters: '14B',
  license: 'MIT',
  task: 'chat',
  source: { host: 'huggingface', repo: 'microsoft/Phi-4-reasoning-onnx', revision: 'f5d8c9c1f0dad99f7d6212eee2c8ea4ebee7b935' },
  architecture: {
    family: 'phi3',
    parameterCount: 14_660_000_000,
    layers: 40,
    hiddenSize: 5120,
    kvHeads: 10,
    headDim: 128,
    vocabSize: 100352,
    maxContextTokens: 32768,
  },
  tokenizer: { path: 'gpu/gpu-int4-rtn-block-32/tokenizer.json', sizeBytes: 7153070, sha256: 'b28a32dcbbb2779573aa8e345457e7b10f470ccfd262d01672c2469698cbd4f7' },
  promptFormat: PHI4_REASONING_FORMAT,
  generation: REASONING_GENERATION,
  quality: {
    rank: 12,
    summary: 'Phi-4 trained to think at length first: top maths and science reasoning here, but slow. A 9 GB download for big GPUs.',
    scores: [
      { benchmark: 'AIME 2024', score: 75.3, source: PHI4_REASONING_CARD },
      { benchmark: 'GPQA Diamond', score: 65.8, source: PHI4_REASONING_CARD },
      { benchmark: 'OmniMath', score: 76.6, source: PHI4_REASONING_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
      engine: 'onnx-webgpu',
      format: 'onnx',
      precision: 'q4f16',
      requiredGpuFeatures: ['shader-f16'],
      graph: { path: 'gpu/gpu-int4-rtn-block-32/model.onnx', sizeBytes: 358825, sha256: 'bc2038958d9a92310c4fc3727bd51b760bc6aa915bd8d9c897f61070ddfdfc93' },
      externalData: [{ path: 'gpu/gpu-int4-rtn-block-32/model.onnx.data', sizeBytes: 8993548288, sha256: '9d2e92e8d7d3cbe31e8613a1f8f884eafbae27bf3bcf2bf9646408665f21fa8b' }],
      kvCacheType: 'float16',
      largestTensorBytes: 1027604480,
      computeOps: 602,
    },
  ],
};

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
  LLAMA32_3B_INSTRUCT,
  PHI35_MINI_INSTRUCT,
  PHI4_MINI_INSTRUCT,
  QWEN3_4B,
  QWEN35_4B,
  DEEPSEEK_R1_QWEN_15B,
  QWEN3_4B_2507,
  GEMMA3_4B_IT,
  SMOLLM2_135M_INSTRUCT,
  GEMMA3_270M_IT,
  LFM25_350M,
  GRANITE4_350M,
  QWEN25_CODER_05B_INSTRUCT,
  LFM2_700M,
  GEMMA3_1B_IT,
  GRANITE4_1B,
  LFM25_12B_INSTRUCT,
  QWEN25_MATH_15B_INSTRUCT,
  LFM2_26B,
  QWEN25_CODER_3B_INSTRUCT,
  GRANITE4_MICRO,
  PHI3_MINI_4K_INSTRUCT,
  OLMO3_7B_INSTRUCT,
  LFM2_8B_A1B,
  LFM2_24B_A2B,
  PHI4_MINI_REASONING,
  MISTRAL_7B_INSTRUCT_V02,
  LLAMA31_8B_INSTRUCT,
  DEEPSEEK_R1_LLAMA_8B,
  GEMMA2_9B_IT,
  PHI3_MEDIUM_4K_INSTRUCT,
  PHI4,
  PHI4_REASONING,
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
export function engineName(_engine: EngineId): string {
  return 'ONNX Runtime';
}

/** Total download size of an artifact (with the tokenizer for ONNX). */
export function artifactBytes(model: ModelSpec, artifact: ModelArtifact): number {
  return model.tokenizer.sizeBytes + weightBytes(artifact);
}

/** Bytes of model weights (what each generated token reads). */
export function weightBytes(artifact: ModelArtifact): number {
  return onnxWeightFiles(artifact).reduce((n, f) => n + f.sizeBytes, 0);
}

/**
 * Identifies exactly which files an artifact runs, so measurements of one file
 * are never applied to another: the graph's SHA-256.
 */
export function artifactFingerprint(artifact: ModelArtifact): string {
  return artifact.graph.sha256;
}

export function artifactUrl(model: Pick<ModelSpec, 'source'>, file: ArtifactFile): string {
  const { repo, revision } = file.from ?? model.source;
  const path = file.from?.path ?? file.path;
  // Development: through the dev server, which serves files already on this Mac (vite.config.ts).
  if (import.meta.env.DEV && globalThis.location) return new URL(`/__models/${repo.replace('/', '__')}/${revision}/${path}`, globalThis.location.origin).href;
  return `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
}
