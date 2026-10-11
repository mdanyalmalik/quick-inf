// The app's IndexedDB database. Each schema version's upgrade step lives in
// UPGRADES; add a new entry (never edit an old one) to change the schema.
//
// Version 1: `artifacts` — one record per downloaded model file (see model-store.ts).
// Version 2: `sessions` — session list metadata (chats today); `messages` — chat
//            messages, indexed by [sessionId, seq] (see src/conversations/).
// Version 3: `runs` — measured model loads and replies, indexed by environment;
//            `deviceTests` — the latest device test per environment (see src/recommend/).
// Version 4: `engineState` — per-engine facts for WebLLM, removed on 2026-10-11; no longer used.

const DB_NAME = 'quick-inf';

const UPGRADES: ((db: IDBDatabase) => void)[] = [
  (db) => {
    db.createObjectStore('artifacts', { keyPath: 'key' });
  },
  (db) => {
    db.createObjectStore('sessions', { keyPath: 'id' });
    const messages = db.createObjectStore('messages', { keyPath: 'id' });
    messages.createIndex('bySession', ['sessionId', 'seq']);
  },
  (db) => {
    const runs = db.createObjectStore('runs', { keyPath: 'id' });
    runs.createIndex('byEnv', 'env');
    db.createObjectStore('deviceTests', { keyPath: 'env' });
  },
  (db) => {
    db.createObjectStore('engineState', { keyPath: 'key' });
  },
];

export type StoreName = 'artifacts' | 'sessions' | 'messages' | 'runs' | 'deviceTests' | 'engineState';

let dbPromise: Promise<IDBDatabase> | null = null;

export function openDb(): Promise<IDBDatabase> {
  dbPromise ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, UPGRADES.length);
    request.onupgradeneeded = (e) => {
      for (let v = e.oldVersion; v < UPGRADES.length; v++) UPGRADES[v]!(request.result);
    };
    request.onsuccess = () => {
      const db = request.result;
      // Another tab upgraded the schema: close so it isn't blocked, and reopen on next use.
      db.onversionchange = () => {
        db.close();
        dbPromise = null;
      };
      resolve(db);
    };
    request.onerror = () => reject(request.error);
    // An older tab still has the database open; it closes on `versionchange` and the upgrade then continues.
    request.onblocked = () => console.warn('Waiting for another Quick Inference tab to release the database.');
  }).catch((err: unknown) => {
    dbPromise = null;
    throw err;
  });
  return dbPromise;
}

function done<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function dbGet<T>(store: StoreName, key: string): Promise<T | undefined> {
  const db = await openDb();
  return done(db.transaction(store).objectStore(store).get(key)) as Promise<T | undefined>;
}

export async function dbGetAll<T>(store: StoreName): Promise<T[]> {
  const db = await openDb();
  return done(db.transaction(store).objectStore(store).getAll()) as Promise<T[]>;
}

/** Writes a record and resolves once the transaction has committed. */
export async function dbPut<T>(store: StoreName, value: T): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(store, 'readwrite', { durability: 'strict' });
  tx.objectStore(store).put(value);
  await committed(tx);
}

export async function dbDelete(store: StoreName, key: string): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(store, 'readwrite');
  tx.objectStore(store).delete(key);
  await committed(tx);
}

function committed(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error('Transaction aborted'));
  });
}

/** All records in `store` whose `index` key lies in `range`, in index order. */
export async function dbGetAllByIndex<T>(store: StoreName, index: string, range: IDBKeyRange): Promise<T[]> {
  const db = await openDb();
  return done(db.transaction(store).objectStore(store).index(index).getAll(range)) as Promise<T[]>;
}

/** Writes several records in one transaction. */
export async function dbPutMany<T>(store: StoreName, values: T[]): Promise<void> {
  if (values.length === 0) return;
  const db = await openDb();
  const tx = db.transaction(store, 'readwrite');
  const os = tx.objectStore(store);
  for (const v of values) os.put(v);
  await committed(tx);
}

/** Deletes several records by key in one transaction. */
export async function dbDeleteMany(store: StoreName, keys: string[]): Promise<void> {
  if (keys.length === 0) return;
  const db = await openDb();
  const tx = db.transaction(store, 'readwrite');
  const os = tx.objectStore(store);
  for (const k of keys) os.delete(k);
  await committed(tx);
}

/**
 * Deletes a session record and all its messages in one transaction, so a
 * crash can't leave messages without their session (or the reverse).
 */
export async function dbDeleteSession(sessionId: string): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(['sessions', 'messages'], 'readwrite');
  tx.objectStore('sessions').delete(sessionId);
  const index = tx.objectStore('messages').index('bySession');
  const request = index.openKeyCursor(sessionRange(sessionId));
  request.onsuccess = () => {
    const cursor = request.result;
    if (!cursor) return;
    tx.objectStore('messages').delete(cursor.primaryKey);
    cursor.continue();
  };
  await committed(tx);
}

/** Every session and message, read in one transaction (a consistent snapshot, for backups). */
export async function dbReadChats<S, M>(): Promise<{ sessions: S[]; messages: M[] }> {
  const db = await openDb();
  const tx = db.transaction(['sessions', 'messages']);
  const [sessions, messages] = await Promise.all([
    done(tx.objectStore('sessions').getAll()) as Promise<S[]>,
    done(tx.objectStore('messages').getAll()) as Promise<M[]>,
  ]);
  return { sessions, messages };
}

/**
 * Adds chats in one transaction: `plan` gets the session ids already saved and
 * returns what to write, so nothing another tab saves in between is overwritten
 * and a failure leaves nothing half-restored.
 */
export async function dbImportChats<S, M, P extends { sessions: S[]; messages: M[] }>(
  plan: (existingIds: Set<string>) => P,
): Promise<P & { hadChats: boolean }> {
  const db = await openDb();
  const tx = db.transaction(['sessions', 'messages'], 'readwrite', { durability: 'strict' });
  const keys = await done(tx.objectStore('sessions').getAllKeys());
  const existing = new Set(keys.map(String));
  const result = plan(existing);
  for (const s of result.sessions) tx.objectStore('sessions').put(s);
  for (const m of result.messages) tx.objectStore('messages').put(m);
  await committed(tx);
  return { ...result, hadChats: existing.size > 0 };
}

/** Index range covering every message of one session. */
export function sessionRange(sessionId: string): IDBKeyRange {
  return IDBKeyRange.bound([sessionId, -Infinity], [sessionId, Infinity]);
}
