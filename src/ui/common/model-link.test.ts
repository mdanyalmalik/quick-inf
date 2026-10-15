import { describe, expect, it } from 'vitest';
import { modelIdFromPath, modelPath } from './model-link';

describe('modelIdFromPath', () => {
  it('reads a model id, with or without the final slash', () => {
    expect(modelIdFromPath('/qwen3-0.6b/', '/')).toBe('qwen3-0.6b');
    expect(modelIdFromPath('/lfm2.5-24b-a2b', '/')).toBe('lfm2.5-24b-a2b');
  });

  it('allows a base path', () => {
    expect(modelIdFromPath('/app/flux.1-schnell/', '/app/')).toBe('flux.1-schnell');
    expect(modelIdFromPath('/other/flux.1-schnell/', '/app/')).toBeNull();
  });

  it('names nothing on the start page or deeper paths', () => {
    expect(modelIdFromPath('/', '/')).toBeNull();
    expect(modelIdFromPath('/a/b/', '/')).toBeNull();
    expect(modelIdFromPath('/%E0%A4%A/', '/')).toBeNull();
  });

  it('round-trips with modelPath', () => {
    for (const id of ['qwen3-0.6b', 'lfm2.5-24b-a2b', 'flux.1-schnell']) expect(modelIdFromPath(modelPath(id, '/'), '/')).toBe(id);
    expect(modelPath(null, '/')).toBe('/');
  });
});
