import { describe, expect, it } from 'vitest';
import { artifactBytes, artifactFingerprint, catalogue, onnxWeightFiles, weightBytes } from './models';

describe('catalogue', () => {
  it('pins every file by size and SHA-256', () => {
    for (const model of catalogue) {
      expect(model.source.revision).toMatch(/^[0-9a-f]{40}$/);
      for (const file of [model.tokenizer, ...model.artifacts.flatMap(onnxWeightFiles)]) {
        expect(file.sizeBytes).toBeGreaterThan(0);
        expect(file.sha256).toMatch(/^[0-9a-f]{64}$/);
      }
      for (const artifact of model.artifacts) {
        expect(artifact.requiredGpuFeatures.includes('shader-f16')).toBe(['q4f16', 'q2f16', 'fp16'].includes(artifact.precision));
        expect(artifactBytes(model, artifact)).toBe(model.tokenizer.sizeBytes + weightBytes(artifact));
        expect(artifactFingerprint(artifact)).toBe(artifact.graph.sha256);
      }
    }
  });

  it('keeps model and artifact ids unique', () => {
    expect(new Set(catalogue.map((m) => m.id)).size).toBe(catalogue.length);
    for (const model of catalogue) expect(new Set(model.artifacts.map((a) => a.id)).size).toBe(model.artifacts.length);
  });
});
