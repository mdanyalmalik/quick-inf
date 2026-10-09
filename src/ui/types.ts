// View-level types for the UI. These describe what the screens need to render,
// not the catalogue/store records that will eventually back them. When the
// model catalogue and persistence modules exist, map their records into these.

/**
 * What a model is used for. Each task gets its own session list and workspace
 * via the registry in `workspaces/registry.ts`; adding a member here is a type
 * error until a matching workspace is registered.
 */
export type ModelTask = 'chat';

export type ModelAvailability = 'on-device' | 'available';

export interface ModelEntry {
  id: string;
  name: string;
  publisher: string;
  parameters: string;
  downloadSize: string;
  contextLength: string;
  task: ModelTask;
  availability: ModelAvailability;
}

/** One item in the session panel: a chat today, possibly other things later. */
export interface Session {
  id: string;
  modelId: string;
  title: string;
  updatedAt: number;
}
