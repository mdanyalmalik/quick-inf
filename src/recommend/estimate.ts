// Performance and memory estimates for a model on this device, from the quick
// device test (src/device/device-test.ts) and the model's size and shape.
//
// Generating one token costs a fixed part (each graph operation is a GPU kernel
// with a CPU/driver cost to issue, plus reading the result back) and the time to
// read all the weights, which are mostly 4-bit:
//
//   decode ms ≈ compute ops × costPerStep × step µs / 1000 + round trip
//             + weight bytes ÷ (4-bit matvec rate × matvecEfficiency)
//             + KV bytes ÷ (bandwidth × bandwidthEfficiency)
//
// Processing a prompt is arithmetic (2 × parameters per token), done in chunks
// that each read the weights again, through slower many-token 4-bit kernels:
//
//   prefill ms ≈ 2 × parameters × tokens ÷ (compute × computeEfficiency)
//              + chunks × chunkWeightReads × weight bytes ÷ 4-bit matvec rate
//
// Device tests from before version 2 lack the 4-bit matvec and step results;
// they use the older raw-bandwidth formula (LEGACY_CALIBRATION).
//
// The constants are calibrated against real runs (see DEFAULT_CALIBRATION);
// measured results always replace estimates in the UI.

import { weightBytes, type ModelArtifact, type ModelSpec } from '../catalogue/models';
import type { DeviceTestResult } from '../device/device-test';
import { prefillChunkTokens } from '../runtime/onnx/limits';

export interface Calibration {
  /** Cost of one graph operation, in units of the test's wall-clock step cost. */
  costPerStep: number;
  /** The runtime's weight-reading rate relative to the test's 4-bit matvec. */
  matvecEfficiency: number;
  /** Fraction of the test's raw read bandwidth reached when reading the KV cache. */
  bandwidthEfficiency: number;
  /** Prompt-processing throughput relative to the test's matrix multiply (above 1: the runtime's kernels beat the test's simple one). */
  computeEfficiency: number;
  /** Each prompt chunk costs this many weight reads at the test's 4-bit matvec rate. */
  chunkWeightReads: number;
}

/**
 * Fitted 2026-10-09 to the four catalogue models on two machines: the owner's Mac
 * (Apple GPU) and a Galaxy S24 Ultra (Adreno 750), Chrome 155, ONNX Runtime Web
 * 1.30, using phone-suite.html (3 timed runs per model). Generation speed: mean
 * error 13%, worst +33% (the two Qwen models on the phone); the previous
 * raw-bandwidth formula was 76% mean and 80–178% too optimistic on the phone.
 * Time to the first word of a 1,500-token prompt: mean 13%, worst −32% (Qwen2.5
 * on the phone). See info/gameplan.md.
 */
export const DEFAULT_CALIBRATION: Calibration = {
  costPerStep: 3.79,
  matvecEfficiency: 0.667,
  bandwidthEfficiency: 0.81,
  computeEfficiency: 3.97,
  chunkWeightReads: 20.2,
};

/**
 * WebLLM's compiled models: a fixed cost per token plus reading the weights;
 * prompts in 1,024-token pieces at a higher share of peak compute than ONNX Runtime.
 * PROVISIONAL, fitted on the Mac only (2026-10-09, decode within ~20%, Qwen worst);
 * refit when phone measurements exist.
 */
export const WEBLLM_CALIBRATION = {
  /** Fixed per-token cost, in units of the device test's per-job step time. */
  fixedSteps: 1900,
  matvecEfficiency: 1.0,
  computeEfficiency: 1.8,
  prefillChunkTokens: 1024,
};

/** For version-1 device tests: dispatch µs instead of step µs, raw bandwidth instead of matvec. */
const LEGACY_CALIBRATION = { bandwidthEfficiency: 0.81, costPerOp: 4.7, computeEfficiency: 2.38 };

/** Conversation length used for the headline generation speed. */
export const TYPICAL_CONTEXT_TOKENS = 300;
/** Prompt sizes for the "first word" estimates. */
export const SHORT_PROMPT_TOKENS = 200;
export const LONG_PROMPT_TOKENS = 2000;
/** Working memory beyond weights and KV cache (activations, runtime buffers). */
const OVERHEAD_BYTES = 256 * 1024 * 1024;

export interface ModelEstimate {
  tokensPerSecond: number;
  firstTokenShortMs: number;
  firstTokenLongMs: number;
  /** GPU memory the model needs at its default context length. */
  memoryBytes: number;
}

export function kvBytesPerToken(model: ModelSpec, artifact: ModelArtifact): number {
  const { layers, kvHeads, headDim } = model.architecture;
  const half = artifact.engine === 'webllm' ? artifact.precision === 'q4f16' : artifact.kvCacheType === 'float16';
  return 2 * layers * kvHeads * headDim * (half ? 2 : 4);
}

export function memoryNeeded(model: ModelSpec, artifact: ModelArtifact): number {
  // WebLLM states what each build needs (weights, cache and buffers at its 4K context).
  if (artifact.engine === 'webllm') return artifact.build.vramMB * 1024 * 1024;
  const logits = prefillChunkTokens(model.architecture.vocabSize) * model.architecture.vocabSize * 4;
  return weightBytes(artifact) + kvBytesPerToken(model, artifact) * model.generation.contextTokens + logits + OVERHEAD_BYTES;
}

export function estimateModel(
  model: ModelSpec,
  artifact: ModelArtifact,
  test: DeviceTestResult,
  calibration: Calibration = DEFAULT_CALIBRATION,
): ModelEstimate {
  const { parameterCount, vocabSize, hiddenSize } = model.architecture;
  const kvPerToken = kvBytesPerToken(model, artifact);
  const kvBytesPerMs = (test.bandwidthGBps * 1e9 * calibration.bandwidthEfficiency) / 1000;
  const chunk = prefillChunkTokens(vocabSize);
  const chunks = (tokens: number) => Math.ceil(tokens / chunk);

  let decodeMs: (context: number) => number;
  let prefillMs: (tokens: number) => number;
  if (artifact.engine === 'webllm') {
    const cal = WEBLLM_CALIBRATION;
    const matvecBytesPerMs = ((test.matvecGBps ?? test.bandwidthGBps) * 1e9) / 1000;
    const stepUs = test.stepUs ?? test.dispatchUs;
    const fixedMs = (cal.fixedSteps * stepUs) / 1000 + test.roundTripMs;
    const weightMs = weightBytes(artifact) / (matvecBytesPerMs * cal.matvecEfficiency);
    decodeMs = (context) => fixedMs + weightMs + (kvPerToken * context) / kvBytesPerMs;
    const flopsPerMs = (test.computeGflops * 1e9 * cal.computeEfficiency) / 1000;
    prefillMs = (tokens) => (2 * parameterCount * tokens) / flopsPerMs + Math.ceil(tokens / cal.prefillChunkTokens) * (fixedMs + weightMs);
  } else if (test.matvecGBps && test.stepUs !== undefined) {
    const fixedMs = (artifact.computeOps * calibration.costPerStep * test.stepUs) / 1000 + test.roundTripMs;
    const matvecBytesPerMs = (test.matvecGBps * 1e9) / 1000;
    const weightMs = weightBytes(artifact) / (matvecBytesPerMs * calibration.matvecEfficiency);
    decodeMs = (context) => fixedMs + weightMs + (kvPerToken * context) / kvBytesPerMs;
    const flopsPerMs = (test.computeGflops * 1e9 * calibration.computeEfficiency) / 1000;
    const chunkMs = (calibration.chunkWeightReads * weightBytes(artifact)) / matvecBytesPerMs;
    prefillMs = (tokens) => (2 * parameterCount * tokens) / flopsPerMs + chunks(tokens) * chunkMs;
  } else {
    const legacy = LEGACY_CALIBRATION;
    const bytesPerMs = (test.bandwidthGBps * 1e9 * legacy.bandwidthEfficiency) / 1000;
    const fixedMs = (artifact.computeOps * legacy.costPerOp * test.dispatchUs) / 1000 + test.roundTripMs;
    const weightMs = weightBytes(artifact) / bytesPerMs;
    decodeMs = (context) => fixedMs + weightMs + (kvPerToken * context) / bytesPerMs;
    const flopsPerMs = (test.computeGflops * 1e9 * legacy.computeEfficiency * (hiddenSize / 1024)) / 1000;
    prefillMs = (tokens) => (2 * parameterCount * tokens) / flopsPerMs + chunks(tokens) * (fixedMs + weightMs);
  }

  return {
    tokensPerSecond: 1000 / decodeMs(TYPICAL_CONTEXT_TOKENS),
    firstTokenShortMs: prefillMs(SHORT_PROMPT_TOKENS),
    firstTokenLongMs: prefillMs(LONG_PROMPT_TOKENS),
    memoryBytes: memoryNeeded(model, artifact),
  };
}
