// Records stored in IndexedDB for conversations (schema v2, see src/store/db.ts).

import type { FinishReason, GenerationStats, RuntimeErrorCode } from '../runtime/contract';

/** One item in a model's session list (a chat today; other kinds later). Metadata only, no messages. */
export interface SessionRecord {
  id: string;
  modelId: string;
  title: string;
  createdAt: number;
  /** Last activity; the session list is sorted by this. */
  updatedAt: number;
}

export type MessageRole = 'user' | 'assistant';

export interface MessageRecord {
  id: string;
  sessionId: string;
  /** Position within the session (ascending). */
  seq: number;
  role: MessageRole;
  content: string;
  /** Assistant messages only. */
  status?: 'streaming' | 'done' | 'error';
  finishReason?: FinishReason;
  /** The reply was still streaming when its tab closed or crashed; `content` is what was saved. */
  interrupted?: boolean;
  stats?: GenerationStats;
  error?: string;
  errorCode?: RuntimeErrorCode;
  createdAt: number;
  updatedAt: number;
}
