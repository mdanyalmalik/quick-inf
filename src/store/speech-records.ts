// Where a speech tab's clip record lives: localStorage, one key per session (the audio itself is a WAV in
// IndexedDB `speech`). Used by the speech tab (src/ui/workspaces/speech/speech-store.ts).

export const speechRecordKey = (sessionId: string) => `quick-inf:speech-session:${sessionId}`;

/** The saved record, as stored (unchecked), or null. */
export function readSpeechRecord(sessionId: string): unknown {
  try {
    return JSON.parse(localStorage.getItem(speechRecordKey(sessionId)) ?? 'null');
  } catch {
    return null;
  }
}

export function writeSpeechRecord(sessionId: string, record: unknown): void {
  localStorage.setItem(speechRecordKey(sessionId), JSON.stringify(record));
}

/** A saved clip (IndexedDB `speech`). */
export interface SpeechRow {
  sessionId: string;
  clipId: string;
  wav: Blob;
}
