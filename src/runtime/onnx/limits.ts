// Sizing rules shared by the ONNX adapter and the performance estimates.

/** Most prompt tokens per forward pass during prefill. */
const MAX_PREFILL_CHUNK = 256;
/** Otherwise each pass returns logits for every token in it (chunk × vocab × 4 bytes); keep that bounded. */
const LOGITS_BUDGET_BYTES = 64 * 1024 * 1024;

/** Prompt tokens processed per forward pass for a model with this vocabulary. */
export function prefillChunkTokens(vocabSize: number, lastTokenLogits = true): number {
  // When the output layer computes only the last position (see external-weights.ts), each pass returns one row.
  if (lastTokenLogits) return MAX_PREFILL_CHUNK;
  return Math.max(16, Math.min(MAX_PREFILL_CHUNK, Math.floor(LOGITS_BUDGET_BYTES / (vocabSize * 4))));
}
