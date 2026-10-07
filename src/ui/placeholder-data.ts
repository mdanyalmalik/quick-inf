// Dummy data so the UI can be built before the catalogue, model store and
// persistence modules exist. Model names are illustrative only: sizes and
// context lengths are not measured or verified, and no model here has been
// tested with this app. Delete this file once real data sources are wired up.

import type { ModelEntry, Session } from './types';
import type { ChatMessage } from './workspaces/chat/types';

export const placeholderModels: ModelEntry[] = [
  {
    id: 'qwen2.5-0.5b-instruct',
    name: 'Qwen2.5 0.5B Instruct',
    publisher: 'Qwen',
    parameters: '0.5B',
    downloadSize: '0.5 GB',
    contextLength: '32K',
    task: 'chat',
    availability: 'on-device',
    recommended: true,
  },
  {
    id: 'llama-3.2-1b-instruct',
    name: 'Llama 3.2 1B Instruct',
    publisher: 'Meta',
    parameters: '1B',
    downloadSize: '1.2 GB',
    contextLength: '128K',
    task: 'chat',
    availability: 'on-device',
  },
  {
    id: 'smollm2-1.7b-instruct',
    name: 'SmolLM2 1.7B Instruct',
    publisher: 'Hugging Face',
    parameters: '1.7B',
    downloadSize: '1.1 GB',
    contextLength: '8K',
    task: 'chat',
    availability: 'available',
  },
  {
    id: 'gemma-2-2b-it',
    name: 'Gemma 2 2B Instruct',
    publisher: 'Google',
    parameters: '2B',
    downloadSize: '1.6 GB',
    contextLength: '8K',
    task: 'chat',
    availability: 'available',
  },
  {
    id: 'phi-3.5-mini-instruct',
    name: 'Phi-3.5 mini Instruct',
    publisher: 'Microsoft',
    parameters: '3.8B',
    downloadSize: '2.2 GB',
    contextLength: '128K',
    task: 'chat',
    availability: 'available',
  },
];

const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;
const now = Date.now();

export const placeholderSessions: Session[] = [
  { id: 's1', modelId: 'qwen2.5-0.5b-instruct', title: 'Rewrite cover letter intro', updatedAt: now - 2 * HOUR },
  { id: 's2', modelId: 'qwen2.5-0.5b-instruct', title: 'Explain JS closures', updatedAt: now - 5 * HOUR },
  { id: 's3', modelId: 'qwen2.5-0.5b-instruct', title: 'Regex for email validation', updatedAt: now - 1 * DAY - 3 * HOUR },
  { id: 's4', modelId: 'qwen2.5-0.5b-instruct', title: 'Weekend trip packing list', updatedAt: now - 4 * DAY },
  { id: 's5', modelId: 'qwen2.5-0.5b-instruct', title: 'Summarise meeting notes', updatedAt: now - 12 * DAY },
  { id: 's6', modelId: 'llama-3.2-1b-instruct', title: 'Python list comprehension', updatedAt: now - 30 * 60 * 1000 },
  { id: 's7', modelId: 'llama-3.2-1b-instruct', title: 'Short poem about autumn', updatedAt: now - 3 * DAY },
];

const defaultConversation: ChatMessage[] = [
  { id: 'm1', role: 'user', content: 'Can you explain what a closure is in JavaScript?' },
  {
    id: 'm2',
    role: 'assistant',
    content:
      'A closure is a function that keeps access to variables from the scope where it was created, even after that scope has finished running.\n\nfunction counter() {\n  let count = 0;\n  return () => ++count;\n}\n\nconst next = counter();\nnext(); // 1\nnext(); // 2\n\nThe returned arrow function "closes over" `count`, so each call sees and updates the same variable.',
  },
  { id: 'm3', role: 'user', content: 'Why is that useful?' },
  {
    id: 'm4',
    role: 'assistant',
    content:
      'It lets you keep private state without globals or classes. Common uses are counters and caches, event handlers that remember configuration, and factory functions that produce pre-configured helpers.',
  },
];

export function placeholderMessagesFor(sessionId: string): ChatMessage[] {
  // Fresh sessions created in the UI start empty; seeded ones share a sample.
  return placeholderSessions.some((s) => s.id === sessionId) ? defaultConversation : [];
}
