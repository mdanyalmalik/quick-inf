// Model catalogue: stable model identities, separate from the artifacts that
// execute them. One model revision can have several artifacts (precisions, and
// later other engines' formats); the runtime picks one the device can run.

export type Precision = 'q4f16' | 'q4';

export interface ArtifactFile {
  /** Path within the pinned repository revision. */
  path: string;
  sizeBytes: number;
  sha256: string;
}

export interface ModelArtifact {
  id: string;
  format: 'onnx';
  precision: Precision;
  /** WebGPU features the artifact needs (e.g. fp16 shaders). */
  requiredGpuFeatures: GPUFeatureName[];
  graph: ArtifactFile;
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

export interface ModelArchitecture {
  family: 'llama' | 'qwen2' | 'qwen3';
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
  source: { host: 'huggingface'; repo: string; revision: string };
  architecture: ModelArchitecture;
  tokenizer: ArtifactFile;
  promptFormat: {
    kind: 'chatml';
    defaultSystemPrompt?: string;
    stopTokens: string[];
    /** Qwen3: answer without a reasoning phase (the template's enable_thinking=false). */
    disableThinking?: boolean;
  };
  generation: GenerationDefaults;
  quality: ModelQuality;
  /** Preferred first. */
  artifacts: ModelArtifact[];
}

const SMOLLM2_CARD = 'SmolLM2 model card (Hugging Face)';
const QWEN25_BLOG = 'Qwen2.5 announcement (Qwen team blog)';
const QWEN3_REPORT = 'Qwen3 technical report, table 20 (thinking off)';
const CHATML_STOPS = ['<|im_end|>', '<|endoftext|>'];

export const SMOLLM2_360M_INSTRUCT: ModelSpec = {
  id: 'smollm2-360m-instruct',
  name: 'SmolLM2 360M Instruct',
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
  name: 'Qwen2.5 0.5B Instruct',
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
  name: 'SmolLM2 1.7B Instruct',
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
    summary: 'The largest here: clearly better at following instructions, knowledge and maths. Slower, and a 1.1 GB download.',
    scores: [
      { benchmark: 'IFEval (prompt/instruction average)', score: 56.7, source: SMOLLM2_CARD },
      { benchmark: 'MT-Bench', score: 6.13, source: SMOLLM2_CARD },
      { benchmark: 'GSM8K (5-shot)', score: 48.2, source: SMOLLM2_CARD },
    ],
  },
  artifacts: [
    {
      id: 'onnx-q4f16',
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

export const catalogue: readonly ModelSpec[] = [
  SMOLLM2_360M_INSTRUCT,
  QWEN25_05B_INSTRUCT,
  QWEN3_06B,
  SMOLLM2_17B_INSTRUCT,
];

export function findModel(id: string): ModelSpec | undefined {
  return catalogue.find((m) => m.id === id);
}

export function artifactUrl(model: ModelSpec, file: ArtifactFile): string {
  const { repo, revision } = model.source;
  return `https://huggingface.co/${repo}/resolve/${revision}/${file.path}`;
}
