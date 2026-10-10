// Which models and engines each kind of device gets.
//
// For now, phones and tablets run only the app's own ONNX engine, and list only the models
// it has files for. On a Galaxy S24 (2026-10-10), WebLLM's prompt steps (1,024 tokens, fixed
// when its models are compiled) hung the GPU on long prompts, which on Android blocks WebGPU
// until the browser restarts; the changes that avoided it cost most of WebLLM's speed there.
// The ONNX engine passed every model. Desktops and laptops prefer WebLLM, with the ONNX
// engine as the fallback. To revisit (see info/devlog.md).

import { isMobileDevice } from '../device/mobile';
import { catalogue, type EngineId, type ModelSpec } from './models';

/** Engines this kind of device uses, preferred first. */
export function deviceEngines(mobile = isMobileDevice()): readonly EngineId[] {
  return mobile ? ['onnx-webgpu'] : ['webllm', 'onnx-webgpu'];
}

/** The catalogue as this kind of device sees it: only models (and files) its engines run. */
export function deviceModels(mobile = isMobileDevice()): ModelSpec[] {
  const engines = deviceEngines(mobile);
  return catalogue
    .map((model) => ({ ...model, artifacts: model.artifacts.filter((a) => engines.includes(a.engine)) }))
    .filter((model) => model.artifacts.length > 0);
}
