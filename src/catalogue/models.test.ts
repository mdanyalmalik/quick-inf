import { describe, expect, it } from 'vitest';
import { artifactBytes, artifactFingerprint, catalogue, weightBytes, type WebLlmArtifact } from './models';
import { WEBLLM_FILES } from './webllm-files';

describe('catalogue', () => {
  it('lists a WebLLM build first wherever a model has one, then the ONNX files', () => {
    for (const model of catalogue) {
      const engines = model.artifacts.map((a) => a.engine);
      const firstOnnx = engines.indexOf('onnx-webgpu');
      if (firstOnnx >= 0) expect(engines.slice(firstOnnx).every((e) => e === 'onnx-webgpu')).toBe(true);
      expect(engines[0]).toBe('webllm');
    }
  });

  it('pins every WebLLM build completely', () => {
    for (const model of catalogue) {
      for (const artifact of model.artifacts) {
        if (artifact.engine !== 'webllm') continue;
        const { build } = artifact;
        expect(build.revision).toMatch(/^[0-9a-f]{40}$/);
        expect(build.modelLib.url).toMatch(/^https:\/\/raw\.githubusercontent\.com\/mlc-ai\/binary-mlc-llm-libs\/[0-9a-f]{40}\//);
        expect(build.modelLib.integrity).toMatch(/^sha256-/);
        expect(build.integrity.config).toMatch(/^sha256-/);
        const files = WEBLLM_FILES[artifact.modelId]!;
        const shards = files.filter((f) => f.md5);
        expect(shards.length).toBeGreaterThan(0);
        for (const s of shards) expect(s.md5).toMatch(/^[0-9a-f]{32}$/);
        expect(files.map((f) => f.path)).toEqual(expect.arrayContaining(['mlc-chat-config.json', 'tensor-cache.json']));
        // The totals the page uses agree with the file lists the worker uses.
        expect(build.weightBytes).toBe(shards.reduce((n, f) => n + f.sizeBytes, 0));
        expect(build.downloadBytes).toBe(files.reduce((n, f) => n + f.sizeBytes, 0) + build.modelLib.sizeBytes);
        expect(files).toContainEqual(build.tokenizer);
        // fp16 builds need fp16 shaders (WebLLM's own list omits this for some models).
        expect(artifact.requiredGpuFeatures.includes('shader-f16')).toBe(artifact.precision !== 'q4');
        expect(artifactBytes(model, artifact)).toBeGreaterThan(weightBytes(artifact));
        expect(artifactFingerprint(artifact)).toBe(`${build.repo}@${build.revision}`);
      }
    }
  });

  it('describes WebLLM-only models from their own builds', () => {
    for (const model of catalogue) {
      if (model.artifacts.some((a) => a.engine === 'onnx-webgpu')) continue;
      expect(model.promptFormat.kind).toBe('model-template');
      const main = model.artifacts[0] as WebLlmArtifact;
      expect(model.source).toEqual({ host: 'huggingface', repo: main.build.repo, revision: main.build.revision });
      expect(model.tokenizer.sha256).toMatch(/^[0-9a-f]{64}$/);
      for (const a of model.artifacts as WebLlmArtifact[]) expect(a.build.vocabSize).toBe(model.architecture.vocabSize);
    }
  });

  it('keeps model and artifact ids unique', () => {
    expect(new Set(catalogue.map((m) => m.id)).size).toBe(catalogue.length);
    for (const model of catalogue) expect(new Set(model.artifacts.map((a) => a.id)).size).toBe(model.artifacts.length);
  });
});
