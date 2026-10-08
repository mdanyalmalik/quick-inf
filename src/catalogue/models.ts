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
  /** Element type of the KV-cache tensors in the graph. */
  kvCacheType: 'float16' | 'float32';
}

export interface ModelArchitecture {
  family: 'llama';
  layers: number;
  kvHeads: number;
  headDim: number;
  vocabSize: number;
  maxContextTokens: number;
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
  promptFormat: { kind: 'chatml'; defaultSystemPrompt?: string; stopTokens: string[] };
  generation: GenerationDefaults;
  /** Preferred first. */
  artifacts: ModelArtifact[];
}

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
  architecture: { family: 'llama', layers: 32, kvHeads: 5, headDim: 64, vocabSize: 49152, maxContextTokens: 8192 },
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
    },
  ],
};

export const catalogue: readonly ModelSpec[] = [SMOLLM2_360M_INSTRUCT];

export function findModel(id: string): ModelSpec | undefined {
  return catalogue.find((m) => m.id === id);
}

export function artifactUrl(model: ModelSpec, file: ArtifactFile): string {
  const { repo, revision } = model.source;
  return `https://huggingface.co/${repo}/resolve/${revision}/${file.path}`;
}
