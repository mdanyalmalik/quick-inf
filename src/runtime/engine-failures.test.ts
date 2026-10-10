import { describe, expect, it } from 'vitest';
import { isEngineFailure } from './engine-failures';

describe('isEngineFailure', () => {
  it('counts GPU and driver failures', () => {
    expect(isEngineFailure({ code: 'device-lost', message: 'x' })).toBe(true);
    expect(isEngineFailure({ code: 'validation-failed', message: 'x' })).toBe(true);
    expect(isEngineFailure({ code: 'load-failed', message: 'WebLLM couldn’t load the model: vkQueueSubmit failed with VK_ERROR_DEVICE_LOST' })).toBe(true);
    expect(isEngineFailure({ code: 'load-failed', message: 'GPUPipelineError: CreateComputePipelines failed with VK_ERROR_UNKNOWN' })).toBe(true);
  });

  it('does not count configuration, network or storage problems', () => {
    expect(isEngineFailure({ code: 'load-failed', message: 'WebLLM couldn’t load the model: Only one of context_window_size and sliding_window_size can be positive.' })).toBe(false);
    expect(isEngineFailure({ code: 'download-failed', message: 'Failed to fetch' })).toBe(false);
    expect(isEngineFailure({ code: 'storage-full', message: 'quota' })).toBe(false);
    expect(isEngineFailure({ code: 'cancelled', message: '' })).toBe(false);
  });
});
