import { describe, expect, it } from 'vitest';
import { findModel } from '../catalogue/models';
import type { DeviceReport } from '../device/probe';
import { chooseBuild, isWebKit } from './onnx/ort-env';
import { checkCompatibility, CPU_MAX_WEIGHT_BYTES } from './compatibility';

const NO_GPU: DeviceReport = { webgpu: 'missing-api', userAgent: 'test' };
const GPU = (features: string[]): DeviceReport => ({
  webgpu: 'available',
  userAgent: 'test',
  gpu: { vendor: 'apple', architecture: '', description: '', features, limits: { maxBufferSize: 4e9, maxStorageBufferBindingSize: 4e9 }, isFallbackAdapter: false },
});

describe('checkCompatibility', () => {
  it('runs on the GPU when WebGPU is usable', () => {
    const report = checkCompatibility(findModel('lfm2.5-350m')!, GPU(['shader-f16']));
    expect(report).toMatchObject({ compatible: true, processor: 'gpu', artifact: { id: 'onnx-q4f16' } });
  });

  it('runs small models on the CPU without WebGPU, preferring float32 maths', () => {
    const report = checkCompatibility(findModel('lfm2.5-350m')!, NO_GPU);
    expect(report).toMatchObject({ compatible: true, processor: 'cpu', artifact: { id: 'onnx-q4' } });
    expect(report.reasons.join(' ')).toMatch(/CPU/);
  });

  it('refuses big models on the CPU', () => {
    const report = checkCompatibility(findModel('qwen2.5-7b-instruct')!, NO_GPU);
    expect(report.compatible).toBe(false);
    expect(report.reasons.join(' ')).toMatch(/too big/);
  });

  it('uses the CPU when the browser has no safe GPU path, even with WebGPU present', () => {
    expect(checkCompatibility(findModel('lfm2.5-350m')!, GPU(['shader-f16']), { cpuOnly: true }).processor).toBe('cpu');
  });

  it('keeps the CPU limit within WebAssembly memory', () => {
    expect(CPU_MAX_WEIGHT_BYTES).toBeLessThan(4e9);
  });
});

describe('ONNX Runtime build', () => {
  it('prefers JSPI, avoids Asyncify on WebKit, and goes CPU-only without WebGPU', () => {
    expect(chooseBuild({ jspi: true, webkit: false, webgpu: true })).toBe('jspi');
    expect(chooseBuild({ jspi: true, webkit: true, webgpu: true })).toBe('jspi');
    expect(chooseBuild({ jspi: false, webkit: false, webgpu: true })).toBe('asyncify');
    expect(chooseBuild({ jspi: false, webkit: true, webgpu: true })).toBe('cpu');
    expect(chooseBuild({ jspi: true, webkit: false, webgpu: false })).toBe('cpu');
  });

  it('knows WebKit from Blink', () => {
    expect(isWebKit('Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit (KHTML, like Gecko) Version Safari')).toBe(true);
    expect(isWebKit('Mozilla/5.0 (iPhone; CPU iPhone OS like Mac OS X) AppleWebKit (KHTML, like Gecko) CriOS Mobile Safari')).toBe(true);
    expect(isWebKit('Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit (KHTML, like Gecko) Chrome/1.0 Safari')).toBe(false);
  });
});
