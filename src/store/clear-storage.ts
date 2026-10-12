// "Clear storage" on the home page: delete every saved model, or everything this app stores in the browser.

import { getRuntimeClient } from '../runtime/client';

/** Deletes every saved model's files (chats and settings stay). */
export async function deleteAllModels(modelIds: string[]): Promise<void> {
  const client = getRuntimeClient();
  for (const id of modelIds) await client.deleteModel(id);
}

/**
 * Deletes everything this app stores: model files, chats and other records, settings and caches. The runtime
 * is stopped first (it holds model files open); the caller reloads the page afterwards.
 */
export async function deleteEverything(): Promise<void> {
  getRuntimeClient().shutdown();
  try {
    const root = await navigator.storage.getDirectory();
    for await (const [name] of (root as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
      await root.removeEntry(name, { recursive: true }).catch(() => {});
    }
  } catch {
    // No saved files, or no file storage in this window.
  }
  for (const db of (await indexedDB.databases?.().catch(() => [])) ?? []) {
    if (!db.name) continue;
    // Open connections in this page close themselves on `versionchange`, which lets the deletion through.
    await new Promise<void>((resolve) => {
      const request = indexedDB.deleteDatabase(db.name!);
      request.onsuccess = request.onerror = request.onblocked = () => resolve();
    });
  }
  for (const key of (await caches?.keys().catch(() => [])) ?? []) await caches.delete(key);
  try {
    localStorage.clear();
    sessionStorage.clear();
  } catch {
    // Nothing to clear.
  }
}
