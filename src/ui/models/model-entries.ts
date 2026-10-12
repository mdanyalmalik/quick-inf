import { useMemo } from 'react';
import { artifactBytes, catalogue, type ModelSpec } from '../../catalogue/models';
import { IMAGE_MODELS, imageModelBytes, runsInApp, type ImageModelSpec } from '../../catalogue/image-models';
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
    task: spec.task,
    availability: saved ? 'on-device' : 'available',
    experimental: spec.experimental !== undefined,
    size: sizeOf(spec),
    parameterCount: spec.architecture.parameterCount,
    reasoning: spec.promptFormat.reasoning ?? false,
    preview: false,
    extraSpec: { label: 'Context', value: `${Math.round(spec.generation.contextTokens / 1024)}K` },
  };
}

function toImageEntry(spec: ImageModelSpec, storage: StorageReport): ModelEntry {
  const runs = runsInApp(spec);
  const saved = (storage.models[spec.id]?.completeArtifacts.length ?? 0) > 0;
  return {
    id: spec.id,
    name: spec.name,
    publisher: spec.publisher,
    parameters: spec.parameters,
    downloadSize: runs ? formatBytes(imageModelBytes(spec)) : 'preview',
    task: 'image',
    availability: saved ? 'on-device' : 'available',
    experimental: false,
    size: spec.parameterCount <= 2.1e9 ? 'small' : spec.parameterCount <= 4.5e9 ? 'medium' : 'large',
    parameterCount: spec.parameterCount,
    reasoning: false,
    preview: !runs,
    license: spec.license,
    extraSpec: { label: 'Image size', value: spec.sizeLabel },
    summary: spec.summary,
  };
}

/** Up to 2B: phones and any laptop; 3-4B: most laptops; 7-9B: 16 GB of memory or more; 14B: 32 GB or a big GPU. */
function sizeOf(spec: ModelSpec): ModelSize {
  if (spec.experimental) return 'experimental';
  const params = spec.architecture.parameterCount;
  return params <= 2.1e9 ? 'small' : params <= 4.5e9 ? 'medium' : params <= 10e9 ? 'large' : 'xlarge';
}

/** Catalogue models as the UI shows them, smallest first. */
export function useModelEntries(): ModelEntry[] {
  const storage = useStorage();
  return useMemo(
    () =>
      [...catalogue.map((spec) => toModelEntry(spec, storage)), ...IMAGE_MODELS.map((spec) => toImageEntry(spec, storage))].sort(
        (a, b) => a.parameterCount - b.parameterCount || a.name.localeCompare(b.name),
      ),
    [storage],
  );
}
