import { describe, expect, it } from 'vitest';
import { deviceModels } from './device-catalogue';
import { catalogue } from './models';

describe('device catalogue', () => {
  it('gives desktops every model and every file', () => {
    expect(deviceModels(false)).toEqual(catalogue);
  });

  it('gives phones only the models the ONNX engine runs, with only those files', () => {
    const phone = deviceModels(true);
    expect(phone.length).toBe(catalogue.filter((m) => m.artifacts.some((a) => a.engine === 'onnx-webgpu')).length);
    expect(phone.every((m) => m.artifacts.length > 0 && m.artifacts.every((a) => a.engine === 'onnx-webgpu'))).toBe(true);
    expect(phone.map((m) => m.id)).not.toContain('gemma-2-2b-it');
    expect(phone.map((m) => m.id)).toContain('qwen3-0.6b');
  });
});
