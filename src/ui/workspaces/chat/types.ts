import type { FinishReason, GenerationStats } from '../../../runtime/contract';

export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  /** Assistant messages only. */
  status?: 'streaming' | 'done' | 'error';
  finishReason?: FinishReason;
  stats?: GenerationStats;
  error?: string;
}
