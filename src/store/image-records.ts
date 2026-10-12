// Where an image tab's generation record lives: localStorage, one key per session (the picture itself, for real
// models, is in IndexedDB `images`). Shared by the image tab (src/ui/workspaces/image/image-store.ts) and backups.

export const imageRecordKey = (sessionId: string) => `quick-inf:image-session:${sessionId}`;

/** The saved record, as stored (unchecked), or null. */
export function readImageRecord(sessionId: string): unknown {
  try {
    return JSON.parse(localStorage.getItem(imageRecordKey(sessionId)) ?? 'null');
  } catch {
    return null;
  }
}

export function writeImageRecord(sessionId: string, record: unknown): void {
  localStorage.setItem(imageRecordKey(sessionId), JSON.stringify(record));
}

/** A saved picture (IndexedDB `images`). */
export interface ImageRow {
  sessionId: string;
  generationId: string;
  png: Blob;
}
