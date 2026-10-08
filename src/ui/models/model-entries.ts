import { catalogue, type ModelSpec } from '../../catalogue/models';
import { formatBytes } from '../common/format';
import type { ModelEntry } from '../types';

function toModelEntry(spec: ModelSpec): ModelEntry {
  const preferred = spec.artifacts[0];
  const bytes = spec.tokenizer.sizeBytes + (preferred?.graph.sizeBytes ?? 0);
  return {
    id: spec.id,
    name: spec.name,
    publisher: spec.publisher,
    parameters: spec.parameters,
    downloadSize: formatBytes(bytes),
    contextLength: `${Math.round(spec.generation.contextTokens / 1024)}K`,
    task: spec.task,
    // Nothing is stored on the device until the OPFS model store exists (milestone 3).
    availability: 'available',
  };
}

export const modelEntries: ModelEntry[] = catalogue.map(toModelEntry);
