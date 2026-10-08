import { useMemo } from 'react';
import { catalogue, type ModelSpec } from '../../catalogue/models';
import type { StorageReport } from '../../runtime/worker/protocol';
import { formatBytes } from '../common/format';
import { useStorage } from '../runtime/use-runtime';
import type { ModelEntry } from '../types';

function toModelEntry(spec: ModelSpec, storage: StorageReport): ModelEntry {
  const preferred = spec.artifacts[0];
  const bytes = spec.tokenizer.sizeBytes + (preferred?.graph.sizeBytes ?? 0);
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
  };
}

/** Catalogue models as the UI shows them, grouped by whether they're saved on this device. */
export function useModelEntries(): ModelEntry[] {
  const storage = useStorage();
  return useMemo(() => catalogue.map((spec) => toModelEntry(spec, storage)), [storage]);
}
