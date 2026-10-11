import { describe, expect, it, vi } from 'vitest';
import { applyGpuBudget, parseSimulation } from './simulation';

describe('parseSimulation', () => {
  it('expands presets and parts', () => {
    expect(parseSimulation('phone')).toEqual({ text: 'phone', mobile: true, gpuBudgetBytes: 3e9, minLimits: true, ramGb: 8 });
    expect(parseSimulation('gpu=2.5')).toEqual({ text: 'gpu=2.5', mobile: false, gpuBudgetBytes: 2.5e9, minLimits: false });
    expect(() => parseSimulation('gpu=lots')).toThrow(/Unknown/);
  });
});

describe('applyGpuBudget', () => {
  // Stand-ins for the WebGPU error classes and a device that records what it creates.
  class FakeOom extends Error {}
  class FakeEvent {
    constructor(
      public type: string,
      public init: { error: Error },
    ) {}
  }
  vi.stubGlobal('GPUOutOfMemoryError', FakeOom);
  vi.stubGlobal('GPUUncapturedErrorEvent', FakeEvent);
  const fakeDevice = () => {
    const events: FakeEvent[] = [];
    const scopes: string[] = [];
    const device = {
      createBuffer: (d: GPUBufferDescriptor) => ({ size: d.size, destroyed: false, destroy() { this.destroyed = true; } }),
      pushErrorScope: (f: string) => void scopes.push(f),
      popErrorScope: async () => (scopes.pop(), null),
      dispatchEvent: (e: FakeEvent) => (events.push(e), true),
    };
    return { device: device as unknown as GPUDevice, events };
  };

  it('fails allocations past the budget like an out-of-memory, and counts destroyed buffers as freed', async () => {
    const { device, events } = fakeDevice();
    applyGpuBudget(device, 100);
    const a = device.createBuffer({ size: 60, usage: 0 });
    const b = device.createBuffer({ size: 60, usage: 0 }) as unknown as { destroyed: boolean };
    expect(b.destroyed).toBe(true);
    expect(events[0]?.init.error).toBeInstanceOf(FakeOom);
    expect(() => device.createBuffer({ size: 60, usage: 0, mappedAtCreation: true })).toThrow(RangeError);
    // Inside an out-of-memory scope the error goes to the scope.
    device.pushErrorScope('out-of-memory');
    device.createBuffer({ size: 60, usage: 0 });
    expect(await device.popErrorScope()).toBeInstanceOf(FakeOom);
    expect(events).toHaveLength(1);
    a.destroy();
    const c = device.createBuffer({ size: 60, usage: 0 }) as unknown as { destroyed: boolean };
    expect(c.destroyed).toBe(false);
  });
});
