import type { ModelLicense } from '../catalogue/image-models';
// View-level types for the UI. These describe what the screens need to render,
// not the catalogue/store records that will eventually back them. When the
// model catalogue and persistence modules exist, map their records into these.

/**
 * What a model is used for. Each task gets its own session list and workspace
 * via the registry in `workspaces/registry.ts`; adding a member here is a type
 * error until a matching workspace is registered.
 */
export type ModelTask = 'chat' | 'image';

export type ModelAvailability = 'on-device' | 'available';

export interface ModelEntry {
  id: string;
  name: string;
  publisher: string;
  parameters: string;
  downloadSize: string;
  task: ModelTask;
  availability: ModelAvailability;
  /** Listed without having been tested anywhere (see ModelSpec.experimental). */
  experimental: boolean;
  /** Size group the sidebar lists it under. */
  size: ModelSize;
  /** Total parameters, for ordering by size. */
  parameterCount: number;
  /** Thinks before answering (shown as a separate block). */
  reasoning: boolean;
  /** A preview of a kind of model the app can't run yet (placeholder results). */
  preview: boolean;
  /** The third fact on the overview page: context length for chat, image size for images. */
  extraSpec: { label: string; value: string };
  summary?: string;
  license?: ModelLicense;
}

/** Rough size groups, by what a device needs to run the model well. */
export type ModelSize = 'small' | 'medium' | 'large' | 'xlarge' | 'experimental';

/** One item in the session panel: a chat today, possibly other things later. */
export interface Session {
  id: string;
  modelId: string;
  title: string;
  updatedAt: number;
}
