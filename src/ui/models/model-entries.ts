import { useMemo } from 'react';
import { deviceModels } from '../../catalogue/device-catalogue';
import { artifactBytes, type ModelSpec } from '../../catalogue/models';
import type { StorageReport } from '../../runtime/worker/protocol';
import { formatBytes } from '../common/format';
import { useStorage } from '../runtime/use-runtime';
import type { ModelEntry, ModelSize } from '../types';

function toModelEntry(spec: ModelSpec, storage: StorageReport): ModelEntry {
  const preferred = spec.artifacts[0];
  const bytes = preferred ? artifactBytes(spec, preferred) : 0;
  const saved = (storage.models[spec.id]?.completeArtifacts.length ?? 0) > 0;
  return {
    id: spec.id,
    name: spec.name,
    publisher: spec.publisher,
    parameters: spec.parameters,
    downloadSize: formatBytes(bytes),
    contextLength: `${Math.round(spec.generation.contextTokens / 1024)}K`,
    task: spec.task,
    availability: saved ? 'on-device' : 'available',
    experimental: spec.experimental !== undefined,
    size: sizeOf(spec),
  };
}

/** Up to 2B: phones and any laptop; 3-4B: most laptops; 7B and up: 16 GB of memory or more. */
function sizeOf(spec: ModelSpec): ModelSize {
  if (spec.experimental) return 'experimental';
  const params = spec.architecture.parameterCount;
  return params <= 2.1e9 ? 'small' : params <= 4.5e9 ? 'medium' : 'large';
}

/** Catalogue models as the UI shows them, grouped by whether they're saved on this device. */
export function useModelEntries(): ModelEntry[] {
  const storage = useStorage();
  return useMemo(() => deviceModels().map((spec) => toModelEntry(spec, storage)), [storage]);
}
