import { useMemo } from 'react';
import { artifactBytes, catalogue, type ModelSpec } from '../../catalogue/models';
import { IMAGE_MODELS, imageModelBytes, runsInApp, type ImageModelSpec } from '../../catalogue/image-models';
import { SPEECH_MODELS, speechModelBytes, type SpeechModelSpec } from '../../catalogue/speech-models';
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
    size: spec.experimental ? 'experimental' : sizeOf(bytes),
    parameterCount: spec.architecture.parameterCount,
    reasoning: (spec.promptFormat.reasoning || spec.promptFormat.thinkingSwitch) ?? false,
    thinkingSwitch: spec.promptFormat.thinkingSwitch ?? false,
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
    size: sizeOf(imageModelBytes(spec)),
    parameterCount: spec.parameterCount,
    reasoning: false,
    preview: !runs,
    license: spec.license,
    extraSpec: { label: 'Image size', value: spec.sizeLabel },
    summary: spec.summary,
  };
}

function toSpeechEntry(spec: SpeechModelSpec, storage: StorageReport): ModelEntry {
  const saved = (storage.models[spec.id]?.completeArtifacts.length ?? 0) > 0;
  return {
    id: spec.id,
    name: spec.name,
    publisher: spec.publisher,
    parameters: spec.parameters,
    downloadSize: formatBytes(speechModelBytes(spec)),
    task: 'speech',
    availability: saved ? 'on-device' : 'available',
    experimental: false,
    // Speech models have a tier below the others: up to 150 MB to download.
    size: speechModelBytes(spec) <= 150e6 ? 'tiny' : sizeOf(speechModelBytes(spec)),
    parameterCount: spec.parameterCount,
    reasoning: false,
    preview: false,
    license: spec.license,
    extraSpec: { label: 'Voices', value: String(spec.voices.length) },
    summary: spec.summary,
  };
}

/** Size groups by download size (close to the memory a model needs): up to 1.2 GB, 3 GB, 7 GB, and larger. */
function sizeOf(downloadBytes: number): ModelSize {
  return downloadBytes <= 1.2e9 ? 'small' : downloadBytes <= 3e9 ? 'medium' : downloadBytes <= 7e9 ? 'large' : 'xlarge';
}

/** Catalogue models as the UI shows them, smallest first. */
export function useModelEntries(): ModelEntry[] {
  const storage = useStorage();
  return useMemo(
    () =>
      [
        ...catalogue.map((spec) => toModelEntry(spec, storage)),
        ...IMAGE_MODELS.map((spec) => toImageEntry(spec, storage)),
        ...SPEECH_MODELS.map((spec) => toSpeechEntry(spec, storage)),
      ].sort(
        (a, b) => a.parameterCount - b.parameterCount || a.name.localeCompare(b.name),
      ),
    [storage],
  );
}
